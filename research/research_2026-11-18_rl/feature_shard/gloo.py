"""Execute two complementary row-sharded expert GEMMs and reconstruct a complete routed output across four real Gloo ranks."""
import ast
import argparse
import json
from pathlib import Path
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from transport import distributed_experts,gather_features,restore_original_rank

HERE=Path(__file__).resolve().parent
def tensors(seed,experts,h):
    gen=torch.Generator().manual_seed(seed)
    levels=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.])
    def matrix(n):
        groups=torch.randint(-4,1,(experts,n,h//16,1),generator=gen)
        scales=(2.**groups)*(1+torch.randint(0,8,groups.shape,generator=gen).float()/8)
        q=torch.randint(16,(experts,n,h//16,16),generator=gen)
        return (levels[q]*scales).reshape(experts,n,h).to(torch.bfloat16)
    return matrix(2*h),matrix(h),torch.linspace(.2,.6,experts).to(torch.bfloat16),torch.linspace(.7,1.15,h).to(torch.bfloat16)

def project(rows,ids,w,gs):
    out=rows.new_empty((len(ids),w.shape[1]))
    for e in ids.unique().tolist():
        where=ids==e
        out[where]=((rows[where].float() @ w[e].float().T).to(torch.bfloat16)*gs[e]).to(torch.bfloat16)
    return out

def activation(up):
    a,b=up.chunk(2,-1)
    return (F.silu(a.float())*b.float()).to(torch.bfloat16)

def run(rank,init,tmp,world):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",rank=rank,world_size=world,init_method=init)
    tp=2; ecount=16; h=128
    groups=[dist.new_group([i,i+1]) for i in range(0,world,2)]
    grp=rank//tp; idx=rank%tp
    group=groups[grp]; half=h//tp
    reports=[]
    for seed in (8211,93137):
        up,down,gs,norm=tensors(seed,ecount,h)
        start=grp*(ecount//len(groups));end=start+ecount//len(groups)
        slot=torch.full((ecount+1,),-1,dtype=torch.int32)
        slot[1+start:1+end]=torch.arange(end-start,dtype=torch.int32)
        features=slice(idx*half,(idx+1)*half)
        local_up=torch.cat([up[start:end,features],up[start:end,h+idx*half:h+(idx+1)*half]],dim=1).contiguous()
        local_down=down[start:end,features].contiguous()
        def w13(x,e,_):return activation(project(x,e,local_up,gs[start:end]))
        def w2(x,e,_):return project(x,e,local_down,gs[start:end])
        for pattern in ("mixed","one_group","invalid","repeated"):
            g=torch.Generator().manual_seed(seed+113)
            m=37;k=8
            x=torch.randn((m,h),generator=g).to(torch.bfloat16)
            keys=torch.randn(m,ecount,generator=g)
            ids=keys.topk(k).indices
            if pattern=="one_group":ids[:]=torch.arange(k)
            if pattern=="invalid":ids.fill_(-1)
            if pattern=="repeated":ids[:]=ids[0].clone()
            scores=keys.topk(k).values.softmax(-1)
            local=distributed_experts(x,ids,scores,slot,idx,group,w13,w2,norm[features])
            full=gather_features(local,group)
            # Only leaders contribute after gathering local H shards.
            if idx:full.zero_()
            dist.all_reduce(full)
            dual=distributed_experts(x,ids,scores,slot,idx,group,w13,w2,norm[features],owner_size=ecount//world)
            restored=restore_original_rank(dual,group)
            dist.all_reduce(restored)
            def original_partial(begin,end):
                live=(ids>=begin)&(ids<end)
                route=live.flatten().nonzero().flatten()
                eid=ids.flatten()[route]
                wx=x[torch.div(route,k,rounding_mode="floor")]
                expert_y=project(activation(project(wx,eid,up,gs)),eid,down,gs)
                v=expert_y.float()*norm
                v=v*torch.rsqrt(v.square().mean(-1,keepdim=True)+1e-8)*scores.flatten()[route,None]
                series=torch.zeros(m*k,h)
                series[route]=v
                acc=torch.zeros(m,h)
                for j in range(k):acc+=series.view(m,k,h)[:,j]
                return acc.to(torch.bfloat16)
            native_ep=original_partial(rank*(ecount//world),(rank+1)*(ecount//world))
            dist.all_reduce(native_ep)
            group_reference=original_partial(start,end)
            if idx:group_reference.zero_()
            dist.all_reduce(group_reference)
            metrics={}
            for label,reference in (("parent_ep",native_ep),("same_group",group_reference)):
                delta=full.float()-reference.float()
                relative=float(delta.norm()/reference.float().norm().clamp_min(1e-10))
                metrics[label]=dict(relative_l2=relative,max_abs=float(delta.abs().max()),equal=float((full==reference).float().mean()))
                assert full.isfinite().all() and relative<(.015 if label=="parent_ep" else .001),(seed,pattern,label,relative)
            delta=restored.float()-native_ep.float()
            rel=float(delta.norm()/native_ep.float().norm().clamp_min(1e-10))
            metrics["owner_preserving"]=dict(relative_l2=rel,bit_mismatches=int((restored.view(torch.int16)!=native_ep.view(torch.int16)).sum()),values=restored.numel())
            assert torch.isfinite(restored).all() and rel <= .001,(seed,pattern,rel)
            reports.append(dict(seed=seed,pattern=pattern,**metrics))
    if rank==0:Path(tmp,"out").write_text(json.dumps(reports))
    dist.destroy_process_group()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--ranks",type=int,choices=(2,4,8),default=4);a=p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        mp.start_processes(run,args=(Path(tmp,"init").as_uri(),tmp,a.ranks),nprocs=a.ranks,start_method="fork")
        data=json.loads(Path(tmp,"out").read_text())
        (HERE/f"tp2_ep{a.ranks//2}_vs_parent_ep{a.ranks}.json").write_text(json.dumps(dict(scope="Actual BF16 activation gather and FP32 sum-of-squares over Gloo; synthetic group-16 quantized weights, CPU FP32 matmuls and reduced dims. No speed claim.",ranks=a.ranks,tests=data),indent=2))
        print("cases",len(data),"max relative-L2",max(r["parent_ep"]["relative_l2"] for r in data), "owner-preserving mismatches",sum(r["owner_preserving"]["bit_mismatches"] for r in data))