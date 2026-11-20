"""Compare graph-owned snapshots and run the unchanged incumbent in both timing positions."""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
import torch
from snapshot import snapshot
here=Path(__file__).resolve().parent
root=here.parents[1]
sys.path.insert(0,str(root/"research_2026-11-18_rl_prefill/decode_pages/global_rows"))
from real_layout import load

ARMS={"b1024w4":(1024,4),"b2048w4":(2048,4),"b4096w4":(4096,4),"b4096w8":(4096,8),"b8192w8":(8192,8)}
def main():
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int,required=True)
    p.add_argument("--arm",choices=list(ARMS));args=p.parse_args()
    source_hash=hashlib.sha256((here/"snapshot.py").read_bytes()).hexdigest()
    parent_hash=hashlib.sha256((root/"release_2026-11-18/candidate.diff").read_bytes()).hexdigest()
    assert parent_hash=="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
    original=load("parent_metadata");previous=load("window_metadata")
    arms=[args.arm] if args.arm else list(ARMS)
    names=["baseline",*arms,"previous","control"]
    cap=131072;fc=1893608;sc=1514886;left=2047;dev="cuda"
    rng=torch.Generator(device=dev).manual_seed(args.seed)
    results=dict(agent="root",scope="Single graph-owned metadata update before the original scheduler read-completion fence, including both full and sliding page-table writes. Native Hopper attention is outside the timer.",
        baseline_sha256=parent_hash,kernel_sha256=source_hash,seed=args.seed,active=True,correct=True,complete=False,shapes={})
    file=here/f"gpu_{args.seed}.json"
    def save():file.write_text(json.dumps(results,indent=2))
    save()
    cases=[(1,2048,1,512),(4,8192,8,512),(16,74160,8,512),(32,74160,8,512),
           (32,74160,8,1),(32,74160,1,0),(7,39001,13,512)]
    for b,ctx,n,extent in cases:
        lengths=[ctx] if b==1 else [int(ctx*(.4+.6*i/(b-1))) for i in range(b)]
        assert sum(lengths)+b*n<=fc
        limit=torch.tensor(lengths,device=dev,dtype=torch.int32)
        table=torch.zeros((b+1,cap),device=dev,dtype=torch.int32)
        slots=torch.arange(1,fc+1,device=dev,dtype=torch.int32)
        if extent:
            count=fc//extent*extent
            order=torch.randperm(count//extent,generator=rng,device=dev)
            slots[:count]=slots[:count].view(-1,extent)[order].flatten()
        start=0
        for i,ll in enumerate(lengths):table[i+1,:ll+n]=slots[start:start+ll+n];start+=ll+n
        mapping=torch.zeros(fc+2,device=dev,dtype=torch.int64)
        seq=torch.empty_like(limit)
        req=torch.arange(1,b+1,device=dev)
        delta=0 if n==1 else n
        def refresh(rep):
            req.copy_(1+torch.randperm(b,device=dev,generator=rng))
            seq.copy_((limit[req-1].float()*(.65+.35*torch.rand(b,device=dev,generator=rng))).int())
            cols=seq[:,None]+delta-1-torch.arange(left+n,device=dev)[None,:]
            ids=table[req[:,None],cols.clamp_min(0).long()][cols>=0].long()
            mapping.zero_();mapping[ids]=(torch.arange(ids.numel(),device=dev)+8069*rep)%sc+1
            mapping[-1]=-1
        refresh(0)
        arrays={name:(torch.empty_like(seq),torch.empty(b+1,device=dev,dtype=torch.int32),
            torch.full((b,cap),-911,device=dev,dtype=torch.int32),torch.full((b,cap),-911,device=dev,dtype=torch.int32)) for name in names}
        def run(name):
            l,cu,full,swa=arrays[name]
            a=(l,cu,full,table,req,cap,seq,delta,1,swa,mapping)
            if name in ("baseline","control"):return original(*a)
            if name=="previous":return previous(*a,swa_suffix_tokens=left+n)
            blk,warps=ARMS[name]
            return snapshot(*a,swa_suffix_tokens=left+n,full_block=blk,warps=warps)
        graphs={}
        for name in names:
            run(name)
            with torch.cuda.graph(g:=torch.cuda.CUDAGraph()):run(name)
            graphs[name]=g
        times={name:[] for name in names}
        checked=0
        for i in range(10):
            refresh(i+1)
            interior=names[1:-1]
            j=(i//2)%len(interior)
            sequence=["baseline",*interior[j:],*interior[:j],"control"]
            if i%2:sequence.reverse()
            for name in sequence:
                a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True);a.record()
                for _ in range(6):graphs[name].replay()
                z.record();z.synchronize();times[name].append(a.elapsed_time(z)*1000/6)
            size,cu,full,swa=arrays["baseline"]
            pos=torch.arange(cap,device=dev)[None,:]
            live=pos<size[:,None];suffix=live & (pos>=size[:,None]-left-n)
            for name in arms+["previous","control"]:
                l,c,f,s=arrays[name]
                assert torch.equal(l,size) and torch.equal(c,cu) and torch.equal(f[live],full[live]) and torch.equal(s[suffix],swa[suffix]),(name,i)
            checked+=int(live.sum()+suffix.sum())
        # A scheduler writer runs after the metadata event and before simulated attention.
        # The owning full and SWA snapshots must still hold the original physical addresses.
        witness=arrays[arms[0]]
        reference=(witness[2][live].clone(),witness[3][suffix].clone())
        ready=torch.cuda.Event();changed=torch.cuda.Event();other=torch.cuda.Stream()
        ready.record()
        with torch.cuda.stream(other):
            other.wait_event(ready);table.fill_(0);mapping.zero_();changed.record()
        torch.cuda.current_stream().wait_event(changed)
        assert torch.equal(reference[0],witness[2][live]) and torch.equal(reference[1],witness[3][suffix])
        k=f"b{b}_k{ctx}_q{n}_extent{extent}"
        results["shapes"][k]=dict(times=times,checked_addresses_per_arm=checked,early_writer=True)
        print(k,{a:round(statistics.median(v),1) for a,v in times.items()},flush=True);save()
        del arrays,graphs,g,full,swa,f,s,witness,reference
    results["complete"]=True;save()
if __name__=="__main__":main()
