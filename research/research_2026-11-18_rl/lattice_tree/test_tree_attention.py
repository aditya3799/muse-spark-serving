"""Compare shared-prefix tree attention with a dense FP32 oracle and a linear control."""
import argparse
import json
import math
import os
from pathlib import Path
import torch
from tree_attention import AttentionReplay
HERE=Path(__file__).resolve().parent

def run(dev,b,n,lengths,window):
    g=torch.Generator().manual_seed(8703+window+n)
    cap=sum(lengths)+n*b+1
    dim=64; kh=2; group=8
    def quant(shape):return (torch.randn(shape,generator=g)*.50).to(torch.float8_e4m3fn).view(torch.uint8)
    q=quant((b,n,kh*group,dim))
    k=quant((cap,kh,dim))
    v=quant((cap,kh,dim))
    perm=torch.randperm(cap-1,generator=g).int()+1
    pages=torch.zeros((b,max(lengths)),dtype=torch.int32)
    sl=torch.empty((b,n),dtype=torch.int32)
    start=0
    for req,ctx in enumerate(lengths):
        pages[req,:ctx]=perm[start:start+ctx];start+=ctx
        sl[req]=perm[start:start+n]; start+=n
    args=[t.to(dev) for t in (q,k,v,pages,sl,torch.tensor(lengths,dtype=torch.int32))]
    kernel=AttentionReplay(b,n,max(lengths),dev,window=window,chunk=64)
    reports=[]
    for shape in ("chain","fork"):
        parent=torch.tensor([[0]+[i-1 if shape=="chain" else (i-1)//2 for i in range(1,n)] for _ in range(b)])
        all_paths=[]
        dep=torch.zeros((b,n),dtype=torch.int32);bit=torch.zeros((b,n),dtype=torch.int64)
        oracle=torch.empty_like(q,dtype=torch.float32)
        qf=q.view(torch.float8_e4m3fn).float();kf=k.view(torch.float8_e4m3fn).float();vf=v.view(torch.float8_e4m3fn).float()
        for req,ctx in enumerate(lengths):
            for i in range(n):
                keep=[i]
                while keep[-1]:keep.append(int(parent[req,keep[-1]]))
                keep=keep[::-1]
                dep[req,i]=len(keep)-1
                bit[req,i]=sum(1<<j for j in keep)
                start=max(0,ctx+len(keep)-1-window) if window else 0
                inds=torch.cat((pages[req,min(start,ctx):ctx],sl[req,keep[max(0,start-ctx):]])).long()
                keys=kf[inds].repeat_interleave(group,dim=1)
                val=vf[inds].repeat_interleave(group,dim=1)
                logits=torch.einsum("hd,lhd->hl",qf[req,i],keys)/math.sqrt(dim)
                oracle[req,i]=torch.einsum("hl,lhd->hd",logits.softmax(-1),val)
                all_paths.append(keep)
        depth,bits=dep.to(dev),bit.to(torch.uint32).to(dev)
        result=kernel.run(*args,depth,bits,True).cpu().clone()
        expanded=torch.zeros((b*3+1,args[3].shape[1]),device=dev,dtype=torch.int32)
        rowids=torch.arange(b,device=dev,dtype=torch.int32)*3+1
        expanded[rowids.long()]=args[3]
        direct=kernel.run(*args[:3],expanded,*args[4:],depth,bits,True,rowids)
        assert torch.equal(result,direct.cpu())
        norm=(result-oracle).abs()
        peak=float(norm.max())
        rms=float(norm.square().mean().sqrt())
        assert torch.isfinite(result).all() and peak<.006 and rms<.0016,(peak,rms)
        if shape=="chain":
            linear=kernel.run(*args,depth,bits,False).cpu().clone()
            assert torch.equal(linear,result)
        else:
            # A poisoned sibling is not permitted to influence the other child's output.
            modified=args[2].clone(); modified[sl[0,1].long()]=0
            changed=kernel.run(args[0],args[1],modified,*args[3:],depth,bits,True).cpu()
            for i,keep in enumerate(all_paths[:n]):
                if 1 not in keep:assert torch.equal(result[0,i],changed[0,i])
            assert (result[0,1]!=changed[0,1]).any()
        reports.append(dict(shape=shape,peak_abs=peak,rms_abs=rms,checked=result.numel()))
    return dict(lengths=lengths,window=window,cases=reports)

def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cpu");args=p.parse_args()
    if args.device=="cpu":assert os.environ.get("TRITON_INTERPRET")=="1"
    torch.set_num_threads(1)
    rows=[run(args.device,2,8,[35,85],left) for left in (0,3,32)]
    if args.device=="cuda":rows.extend(run("cuda",2,8,[2048,8193],x) for x in (2048,0))
    (HERE/f"tree_attention_{args.device}.json").write_text(json.dumps(rows,indent=2))
    print(rows)
if __name__=="__main__":main()
