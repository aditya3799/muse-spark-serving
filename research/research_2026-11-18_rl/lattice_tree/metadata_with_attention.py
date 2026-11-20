"""Run the incumbent and direct metadata builders with identical portable full and sliding GQA consumers."""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
import torch
from tree_attention import AttentionReplay
here=Path(__file__).resolve().parent
root=here.parents[1]
global_dir=root/"research_2026-11-18_rl_prefill/decode_pages/global_rows"
sys.path.insert(0,str(global_dir))
from real_layout import load

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--cases",nargs="+",default=["1x2048","4x8192","16x74160","32x74160"])
    parser.add_argument("--layer-stack",action="store_true",help="Replay all 16 full and 46 sliding attention calls per metadata update")
    parser.add_argument("--private-snapshot",action="store_true",help="compare the selected owned-table snapshot against the same earliest parent")
    args=parser.parse_args()
    old,new=load("parent_metadata"),load("metadata")
    fullcap=1893608;swacap=1514886;ctxcap=131072;n=8;left=2047;dev="cuda"
    generator=torch.Generator(device=dev).manual_seed(args.seed)
    def rng(shape):return torch.randint(15,54,shape,dtype=torch.uint8,device=dev,generator=generator)
    k=rng((fullcap+1,2,64));v=rng((fullcap+1,2,64))
    report=dict(agent="root",candidate="Direct FA3 request-row metadata plus a portable FP8-input full and SWA attention read",
        scope="One RTX3050 rank. Includes the actual once-per-forward metadata kernel then one portable full and one portable sliding GQA attention kernel. The portable consumer reads the real FP8 cache format and uses FP16 dot operands on Ampere. Native FA3, the other 60 layers, QKV entry, TP, and scheduler publication are outside this measurement.",
        baseline_sha256=hashlib.sha256((root/"release_2026-11-18/candidate.diff").read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256((global_dir/"candidate.diff").read_bytes()).hexdigest(),
        seed=args.seed,correct=True,active=True,complete=False,bench=dict(instances=1,shapes={}))
    layers=tuple("full" if (61-i)%4==0 else "sliding" for i in range(62)) if args.layer_stack else ("full","sliding")
    if args.layer_stack:
        assert layers.count("full")==16 and layers.count("sliding")==46
        report.update(candidate="Direct page-row metadata with the 62-layer portable attention schedule",
                      scope="One parent or direct metadata build, then the 16 full and 46 sliding paged attention calls in checkpoint order on one RTX 3050. Uses 62 independently changing query inputs and a shared FP8 KV bank to fit memory. Omits attention projection and post-processing, expert and dense FFNs, eight-rank communication, and publication. Measures index overhead and amortization over the attention stack using portable FP16 dots.")
    stem="metadata_stack" if args.layer_stack else "metadata_attention"
    if args.private_snapshot:
        import re
        snap_dir=here.parent/"decode_snapshot"
        sys.path.insert(0,str(snap_dir))
        from snapshot import snapshot
        manifest=json.loads((snap_dir/"manifest.json").read_text())
        arm=json.loads((snap_dir/"selection.json").read_text())["selected"]
        assert manifest["arm"]==arm
        blk,wp=map(int,re.fullmatch(r"b([0-9]+)w([0-9]+)",arm).groups())
        report.update(candidate="Private full/SWA page snapshots with "+("62-layer" if args.layer_stack else "two-layer")+" portable paged attention",
                      candidate_sha256=manifest["candidate_sha256"],variant_identity=arm)
        report["scope"]+=" The candidate keeps privately materialized tables and original early publication."
        stem+="_private"
    outfile=here/f"{stem}_{args.seed}.json"
    def save():outfile.write_text(json.dumps(report,indent=2))
    for case in args.cases:
        b,length=map(int,case.split("x"));lens0=[length if b==1 else int(length*(.4+.6*i/(b-1))) for i in range(b)]
        assert sum(lens0)+b*n<=fullcap
        table=torch.zeros((b+1,ctxcap),device=dev,dtype=torch.int32)
        extent=512
        count=(fullcap//extent)*extent
        slots=torch.arange(1,fullcap+1,device=dev,dtype=torch.int32)
        permutation=torch.randperm(count//extent,device=dev,generator=generator)
        slots[:count]=slots[:count].view(-1,extent)[permutation].reshape(-1)
        offset=0
        for i,ln in enumerate(lens0):
            table[i+1,:ln+n]=slots[offset:offset+ln+n];offset+=ln+n
        request=torch.arange(1,b+1,device=dev,dtype=torch.int64)
        limit=torch.tensor(lens0,device=dev,dtype=torch.int32)
        lens=torch.empty_like(limit);cache_idx=torch.empty_like(limit)
        mapping=torch.zeros(fullcap+2,device=dev,dtype=torch.int64)
        q=rng((b,n,16,64));depth=torch.arange(n,device=dev,dtype=torch.int32).expand(b,n).contiguous()
        bits=((torch.ones_like(depth)<< (depth+1))-1).to(torch.uint32)
        queries=rng((len(layers),b,n,16,64)) if args.layer_stack else None
        slot=torch.empty((b,n),dtype=torch.int32,device=dev);swa_slot=torch.empty_like(slot)
        def refresh(rep):
            request.copy_(torch.randperm(b,device=dev,generator=generator)+1)
            lens.copy_((limit[request-1].float()*(.68+.32*torch.rand(b,device=dev,generator=generator))).int())
            cols=lens.long()[:,None]+torch.arange(n,device=dev)[None,:]
            slot.copy_(table[request[:,None],cols])
            suffix=lens[:,None]+n-1-torch.arange(left+n+4,device=dev)[None,:]
            physical=table[request[:,None],suffix.clamp_min(0).long()][suffix>=0].long()
            assert torch.unique(physical).numel()==physical.numel()
            mapping.zero_()
            mapping[physical]=(torch.arange(physical.numel(),device=dev)+rep*8011)%swacap+1
            mapping[-1]=-1
            swa_slot.copy_(mapping[slot.long()].int())
        refresh(0)
        def alloc():
            return (torch.empty_like(limit),torch.empty(b+1,dtype=torch.int32,device=dev),
                    torch.empty((b,ctxcap),dtype=torch.int32,device=dev),
                    torch.full((b,ctxcap),-1,dtype=torch.int32,device=dev))
        full=AttentionReplay(b,n,length,dev,chunk=2048)
        sliding=AttentionReplay(b,n,length,dev,window=left,chunk=2048)
        def run(state,direct,collect=False):
            ll,cu,pages,swa=state
            if direct and args.private_snapshot:
                snapshot(ll,cu,pages,table,request,ctxcap,lens,n,1,swa,mapping,
                         swa_suffix_tokens=left+n,full_block=blk,warps=wp)
                direct=False
            elif direct:
                new(ll,cu,pages,table,request,ctxcap,lens,n,1,swa,mapping,
                    swa_suffix_tokens=left+n,direct_full_indices=cache_idx)
            else:
                old(ll,cu,pages,table,request,ctxcap,lens,n,1,swa,mapping)
            snapshots=[]
            for i,kind in enumerate(layers):
                query=queries[i] if args.layer_stack else q
                if kind=="full":
                    out=full.run(query,k,v,table if direct else pages,slot,lens,depth,bits,False,cache_idx if direct else None)
                else:
                    out=sliding.run(query,k,v,swa,swa_slot,lens,depth,bits,False)
                if collect:snapshots.append(out.clone())
            return snapshots
        arrays={name:alloc() for name in ("baseline","candidate","control")}
        graphs={}
        for name,state in arrays.items():
            run(state,name=="candidate")
            with torch.cuda.graph(graph:=torch.cuda.CUDAGraph()):run(state,name=="candidate")
            graphs[name]=graph
        times={name:[] for name in arrays}
        matched=0
        for rep in range(8):
            refresh(rep+1)
            q.random_(15,54,generator=generator)
            if args.layer_stack:queries.random_(15,54,generator=generator)
            order=["baseline","candidate","control"] if rep%2==0 else ["control","candidate","baseline"]
            for name in order:
                a=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
                a.record();graphs[name].replay();e.record();e.synchronize();times[name].append(a.elapsed_time(e)*1000)
            if args.layer_stack and rep in (0,7):
                refs=run(arrays["baseline"],False,True)
                checks=run(arrays["candidate"],True,True)
                ok=all(torch.equal(a,b0) and bool(torch.isfinite(a).all()) for a,b0 in zip(refs,checks))
                matched+=sum(a.numel() for a in refs)
                del refs,checks
            else:
                graphs["baseline"].replay()
                ref1,ref2=full.out.clone(),sliding.out.clone()
                graphs["candidate"].replay()
                ok=torch.equal(ref1,full.out) and torch.equal(ref2,sliding.out)
                matched+=ref1.numel()+ref2.numel()
            report["correct"] &= ok
            assert ok and torch.isfinite(full.out).all() and torch.isfinite(sliding.out).all(),case
        report["bench"]["shapes"][case]=dict(weight=1,times=times,matched_floats=matched)
        print(case,{a:round(statistics.median(t)) for a,t in times.items()},matched,flush=True)
        save();del full,sliding,graphs,graph,arrays,state
    report["complete"]=True;save()
if __name__=="__main__":main()
