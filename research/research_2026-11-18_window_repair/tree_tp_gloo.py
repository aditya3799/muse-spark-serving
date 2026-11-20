"""Two real Gloo processes with adversarially distinct proposals and target argmax values."""
import ast,json,os,sys,tempfile
from pathlib import Path
from types import SimpleNamespace as N
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
H=Path(__file__).resolve().parent;K=H.parent/"research_2026-11-18_rl/lattice_tree"
sys.path.insert(0,str(K))
from fixed_tree_adapter import TreeStep

def task(rank,path,port):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+path,rank=rank,world_size=2)
    d=np.load(K/"public_pair/trace_05.npz");b=2
    rows=torch.arange(b);lens=torch.tensor([5,9],dtype=torch.int32)
    req=torch.tensor([2,1]);table=torch.arange(128,dtype=torch.int32).reshape(4,32)
    slots=table[req[:,None],lens[:,None]+torch.arange(8)].long()
    group=N(world_size=2,broadcast=lambda tensor,src:dist.broadcast(tensor,src))
    par=N(attn_dp_enabled=False,attn_dcp_size=1,enable_prefill_cp=False)
    env=dict(SIMULATE_ACC_LEN=0,torch=torch,TreeStep=TreeStep,WIDTH=8,get_parallel=lambda:par,get_dp_tp_group=lambda:group,
             SpecTpSyncSite=N(DFLASH_ACCEPT_GREEDY=15),IdSpaceKind=N(SLIDING_WINDOW=2))
    for f in ast.parse((H/"tree_dispatch.py").read_text()).body:
        if isinstance(f,ast.FunctionDef):exec(compile(ast.unparse(f),"adapter","exec"),env)
    be=type("FlashAttentionBackend",(),{})()
    for key,v in dict(fa_impl_ver=3,page_size=1,use_mla=False,fa_skip_kv_cache=False,kv_cache_is_mxfp8=False,
                    has_local_attention=False,use_sliding_window_kv_pool=True,sliding_window_size=4).items():setattr(be,key,v)
    runner=N(attn_backend=be,model=type("MuseSparkTextForCausalLM",(),{})(),
        req_to_token_pool=N(req_to_token=table),kv_index_translator=N(is_translating=False,defer_read_translate=False),
        token_to_kv_pool=N(full_to_swa_index_mapping=(torch.arange(160,dtype=torch.int32)+31*rank)%160))
    w=N(model_runner=runner,lilicorr=None,_need_mamba_verify_commit=False,
        _tp_sync=N(enabled=lambda x:False),_can_skip_verify_logits=lambda batch,sampling_info:True)
    def buffers(bs):
        return (torch.empty(bs,dtype=torch.int32),torch.empty(bs,dtype=torch.int32),torch.empty(bs,dtype=torch.long),
                torch.empty((bs,8),dtype=torch.long),torch.empty(bs,dtype=torch.long))
    w._next_accept_bonus_buffers=buffers
    batch=N(forward_mode=N(is_idle=lambda:False),sampling_info=None,req_pool_indices=req,
        reqs=[N(kv=N(kv_allocated_len=int(l)+8)) for l in lens])
    n=0;nonlinear=0
    for i in (2,11,21,34):
        pos=d["anchors"][i:i+2]
        w._draft_sampler=N(tree_scores=torch.tensor(d["scores"][i:i+2]).float(),
                          candidate_out=torch.tensor(d["ids"][i:i+2]).long()+1000000*rank)
        step=env["select_tree"](w,batch,lens,slots,torch.tensor(d["tokens"][pos]).long(),True)
        target=torch.full((b,8),-1000-rank,dtype=torch.long)
        for k in range(b):
            p=0
            while True:
                c=[j for j in range(1,8) if step.parents[k,j]==p]
                if not c:break
                child=c[-1]
                if child!=p+1:nonlinear+=1
                target[k,p]=step.tokens[k,child]+rank*10000
                p=child
        result=env["accept_tree"](w,step,target,None,batch,lens,slots)
        # Check the complete virtual permutation, token decisions and wire on both processes.
        state=torch.cat([table.flatten().long(),step.wire.long(),*[x.flatten().long() for x in result]])
        others=[torch.empty_like(state) for _ in range(2)];dist.all_gather(others,state)
        assert torch.equal(*others)
        n+=b
    if rank==0:
        (H/"fixed_tree/tp_gloo.json").write_text(json.dumps(dict(ranks=2,requests=n,distinct_local_proposals=True,
            distinct_unreconciled_target_argmax=True,commit_site_was_disabled=True,identical_decisions_and_all_reserved_slots=True,
            nonordinal_edges=nonlinear,scope="Actual Gloo CPU topology and target broadcasts using pinned adapter with Triton interpreter and independent SWA maps; no NCCL timing."),indent=2))
    dist.destroy_process_group()

if __name__=="__main__":
    assert os.environ.get("TRITON_INTERPRET")=="1" and not torch.cuda.is_available()
    with tempfile.TemporaryDirectory() as d:
        mp.start_processes(task,args=(str(Path(d)/"store"),0),nprocs=2,join=True,start_method="fork")

