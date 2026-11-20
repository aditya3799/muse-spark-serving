"""Exercise the production adapter methods over reciprocal TP collectives with Gloo."""
import argparse
import ast
import datetime
import json
import os
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
from typing import Optional
import shutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE=Path(__file__).resolve().parent

def extract(file,names,ns):
    asts=[m for m in ast.walk(ast.parse((HERE/file).read_bytes())) if isinstance(m,ast.FunctionDef) and m.name in names]
    assert len(asts)==len(names)
    exec(compile(ast.Module(asts,[]),file,'exec'),ns)
    return ns

class Group:
    def __init__(self,r,w):
        self.rank_in_group=r; self.world_size=w
    def all_gather(self,x,dim):
        y=[torch.empty_like(x) for _ in range(self.world_size)]
        dist.all_gather(y,x.contiguous()); return torch.cat(y,dim=dim)
    def all_gather_into_tensor(self,y,x):
        dist.all_gather_into_tensor(y,x)
    def reduce_scatter_tensor(self,y,x):
        dist.reduce_scatter_tensor(y,x.contiguous())

class Math:
    float32=torch.float32
    @staticmethod
    def mm(a,b,out_dtype):
        assert a.dtype==b.dtype==torch.bfloat16 and out_dtype==torch.float32
        return torch.mm(a.float(),b.float())

def linear(x,w):
    return (x.float() @ w.float().T).to(torch.bfloat16)

class Experts:
    use_humming=True
    post_expert_norm=True
    def __init__(self,rank):
        self.rank=rank
    def __call__(self,x,scores,ids,post_expert_norm,pre_expert_norm):
        assert post_expert_norm and pre_expert_norm
        z=(x.float()*torch.rsqrt(x.float().square().mean(1,keepdim=True)+1e-6)).to(x.dtype)
        sel=(ids==self.rank)
        return (scores.mul(sel).sum(1,keepdim=True)*torch.tanh(z.float()*(1+self.rank/8))).to(x.dtype)

def go(rank,world,url,variant):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method=url,rank=rank,world_size=world,timeout=datetime.timedelta(seconds=180))
    g=Group(rank,world)
    parallel=SimpleNamespace(tp_group=g)
    ns=dict(torch=torch,Optional=Optional,nn=torch.nn,get_parallel=lambda:parallel,
            ceil_align=lambda m,k:(m+k-1)//k*k,sp_fused_matmul_eligible=lambda linear:False)
    ns["set_sp_num_tokens"]=lambda n:ns.update(_rows=n)
    ns["sp_num_tokens"]=lambda:ns["_rows"]
    extract("layernorm_sp.py",["sp_entry_scatter","sp_exit_gather","row_parallel_gbar_matmul","column_parallel_g_matmul"],ns)
    extract(("experimental_local/" if variant=="local" else "")+"muse_spark.py",["_forward_sp_experts"],ns)
    proj=extract("muse_spark_gemm_ag.py",["local_prefill_projection","prefill_row_projection"],{"torch":Math})
    experts=Experts(rank); leg=SimpleNamespace(experts=experts,pre_expert_norm=True)
    accum=[]
    def record(name,x,y,case):
        actual=int(torch.count_nonzero(x!=y))
        case[name]=[actual,x.numel(),float((x.float()-y.float()).abs().max())]
    def boundary(r,b):
        n=b.float()*torch.rsqrt(b.float().square().mean(1,keepdim=True)+1e-6)
        stream=r+n*0.67
        return (stream*torch.rsqrt(stream.square().mean(1,keepdim=True)+1e-6)).to(b.dtype),stream
    for seed,rows,style in ((1741,512,"random"),(3499,520,"random"),(8327,16384,"sparse"),(8329,1024,"cancellation")):
        gen=torch.Generator().manual_seed(seed)
        def rand(*s):return torch.randn(s,generator=gen).to(torch.bfloat16)
        h,e,k=64,32,8
        stream=rand(rows,h).float(); x=rand(rows,k)
        pool=rand(e,h);unpool=rand(h,e);qkv=rand(k,h);router=rand(world,h).float()
        if style=="sparse":x[:,1:]=0
        rank_source=torch.Generator().manual_seed(seed*101+rank)
        o=torch.randn((h,k),generator=rank_source).to(torch.bfloat16)
        if style=="cancellation":
            shared_o=rand(h,k)
            o=(shared_o*(1 if rank%2==0 else -1)+0.00390625*o).to(torch.bfloat16)
        cls=SimpleNamespace(tp_group=g,weight=o,quant_method=SimpleNamespace(apply=lambda l,z,b:linear(z,l.weight)))
        local=ns["sp_entry_scatter"](stream)
        ref_o=linear(x,o);dist.all_reduce(ref_o)
        cand_o=ns["row_parallel_gbar_matmul"](cls,x,None)
        ref_input,ref_stream=boundary(stream,ref_o);cand_input,cand_stream=boundary(local,cand_o)
        ref_pool=proj["prefill_row_projection"](g,ref_input,pool)
        cand_pool=proj["local_prefill_projection"](cand_input,pool)
        def routes(b):
            vals,ids=torch.topk((b.float()@router.T),min(2,world),dim=-1)
            return vals.softmax(-1),ids.int()
        sr,ir=routes(ref_input);sc,ic=routes(cand_input)
        ref_e=experts(ref_pool,sr,ir,True,True);dist.all_reduce(ref_e)
        if variant=="gather":
            sc,ic=routes(g.all_gather(cand_input,0))
        cand_e=ns["_forward_sp_experts"](leg,cand_pool,sc,ic,None)
        ref_unpool=proj["prefill_row_projection"](g,ref_e,unpool)
        cand_unpool=proj["local_prefill_projection"](cand_e,unpool)
        rn,rs=boundary(ref_stream,ref_unpool);cn,cs=boundary(cand_stream,cand_unpool)
        cls.weight=qkv
        next_q=ns["column_parallel_g_matmul"](cls,cn,None)
        j=dict(seed=seed,rows=rows,fixture=style)
        for name,a,b in (("o_reduction",ref_o,cand_o),("pooled",ref_pool,cand_pool),
                         ("expert_reduction",ref_e,cand_e),("unpool",ref_unpool,cand_unpool),("residual",rs,cs)):
            record(name,a,g.all_gather(b,0),j)
        record("router_ids",ir,g.all_gather(ic,0) if variant=="local" else ic,j)
        record("route_weights",sr,g.all_gather(sc,0) if variant=="local" else sc,j)
        record("next_qkv",linear(rn,qkv),next_q,j)
        taps=[(cs+i).to(torch.bfloat16) for i in range(5)]
        record("packed_five_taps",torch.cat([(rs+i).to(torch.bfloat16) for i in range(5)],dim=-1),
               ns["sp_exit_gather"](torch.cat(taps,dim=-1),rows),j)
        if rank==0: accum.append(j)
    if rank==0:
        (HERE/f"gloo_{variant}_tp{world}.json").write_text(json.dumps(accum,indent=2))
        print(world,[(c["rows"],c["router_ids"][0],c["residual"][0],c["next_qkv"][0]) for c in accum],flush=True)
    dist.destroy_process_group()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--ranks",required=True,type=int)
    parser.add_argument("--variant",choices=("local","gather"),default="gather")
    arg=parser.parse_args()
    os.environ["USE_LIBUV"]="0"
    with tempfile.TemporaryDirectory(dir=HERE) as t:
        mp.start_processes(go,args=(arg.ranks,(Path(t)/"store").as_uri(),arg.variant),nprocs=arg.ranks,join=True,
                           start_method="spawn" if sys.platform=="win32" else "fork")

