"""One exact values/IDs collective following the original local top-k, with eight live CPU ranks."""
from pathlib import Path
import hashlib,json,os,tempfile,time,statistics
import torch,torch.distributed as dist
H=Path(__file__).resolve().parent;D=H/"decode_selector";WORLD=8;K=16
def gathered(x):
    n,m=x.shape
    out=torch.empty((WORLD*n,m),dtype=x.dtype)
    dist.all_gather_into_tensor(out,x)
    return out.view(WORLD,n,m).movedim(0,1).reshape(n,WORLD*m)
def base(x,idx):
    vals=gathered(x.float());ids=gathered(idx.long())
    top,s=torch.topk(vals,K,dim=-1)
    return torch.gather(ids,-1,s),top
def pack(x,idx):
    bits=(x.float().view(torch.int32).to(torch.int64)<<32)|idx.to(torch.int64)
    allbits=gathered(bits)
    vals=(allbits>>32).int().view(torch.float32)
    top,s=torch.topk(vals,K,dim=-1)
    return torch.gather(allbits,-1,s).bitwise_and_(0xffffffff),top
def rankbench(rank,init,shared,seed):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+init,rank=rank,world_size=WORLD)
    g=torch.Generator().manual_seed(seed+rank*831)
    res={}
    for b in (1,4,16,32):
        n=7*b;shard=25256
        tx={t:[] for t in ("baseline","candidate","control")};wrongs=0
        for step in range(14):
            x=torch.randn((n,K),generator=g).bfloat16()
            ix=torch.randint(shard,(n,K),generator=g,dtype=torch.int64)+rank*shard
            if step%5==0:x[:,0]=float("nan")
            if step%3==0:x[:,1:4]=2.
            if step%7==0:x[:,6:8]=float("-inf")
            order=["baseline","candidate","control"]
            k=(step//2)%3;order=order[k:]+order[:k]
            if step%2:order.reverse()
            outputs={}
            for arm in order:
                dist.barrier()
                t=time.perf_counter_ns()
                outputs[arm]=(pack if arm=="candidate" else base)(x,ix)
                elapsed=(time.perf_counter_ns()-t)/1000
                # The collective is rate limited by the slowest real participant.
                wall=torch.tensor(elapsed,dtype=torch.float64)
                dist.all_reduce(wall,op=dist.ReduceOp.MAX)
                if step>=2:tx[arm].append(wall.item())
            for arm in ("candidate","control"):
                ids,v=outputs[arm];base_ids,base_v=outputs["baseline"]
                wrongs+=int((ids!=base_ids).sum())+int((v.view(torch.int32)!=base_v.view(torch.int32)).sum())
        res[str(b)]=dict(times=tx,wrongs=wrongs,bytes_per_rank={"baseline":n*K*12,"candidate":n*K*8},collectives={"baseline":2,"candidate":1})
    w=torch.tensor(sum(v["wrongs"] for v in res.values()))
    dist.all_reduce(w)
    if rank==0:
        assert w.item()==0
        Path(shared).write_text(json.dumps(dict(seed=seed,gloo_world=8,scope="eight live Gloo CPU ranks, source gather layout, bit-for-bit global top-k including ties, NaNs and infinities, no NCCL or H200",shapes=res),indent=2))
        print({k:{a:round(statistics.median(t),1) for a,t in r["times"].items()} for k,r in res.items()})
    dist.destroy_process_group()
if __name__=="__main__":
    import sys
    label=sys.argv[1];seed=(18373 if label=="pilot" else 851831)
    with tempfile.TemporaryDirectory() as tmp:
        torch.multiprocessing.start_processes(rankbench,args=(str(Path(tmp)/"filestore"),str(D/f"gloo_{label}.json"),seed),nprocs=WORLD,start_method="fork")

