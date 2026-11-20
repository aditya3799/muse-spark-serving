"""Run native Ampere BF16 paged attention with frozen page builders and shared/split scheduling."""
import argparse
import hashlib
import inspect
import json
import statistics
import sys
from pathlib import Path

import torch
import sgl_kernel.flash_attn as fa

here=Path(__file__).resolve().parent
root=here.parents[1]
glob=root/"research_2026-11-18_rl_prefill/decode_pages/global_rows"
sys.path.insert(0,str(glob))
from real_layout import load
pvt=here.parent/"decode_snapshot"
sys.path.insert(0,str(pvt))
from snapshot import snapshot

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--cases",nargs="+",default=["1x2048","4x8192","16x74160","32x74160"])
    p.add_argument("--arms",nargs="+",default=["schedule","direct_schedule","private_schedule"],
                   choices=["schedule","direct","private","direct_schedule","private_schedule"])
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args();a=torch.cuda.current_device()
    assert fa.is_fa3_supported(a)
    baseline=load("parent_metadata");fast=load("metadata")
    wrap=Path(inspect.getfile(fa));wrapper_hash=hashlib.sha256(wrap.read_bytes()).hexdigest()
    assert wrapper_hash=="610747f2c68495c5f3ba67f1247f0a1b4e4f7df116e2f89cd8e8983f0271f9ac"
    cfg=json.loads((pvt/"manifest.json").read_text())
    assert cfg["arm"]=="b2048w4"
    rng=torch.Generator(device="cuda").manual_seed(args.seed)
    cap=131072;fc=1893608;sc=1514886;n=8;left=2047;dev="cuda"
    layers=[i for i in range(62)] if not args.smoke else [0,1,2,3,4,5]
    # Keep a bounded native BF16 K/V bank in VRAM, using each address as a real page.
    k=torch.empty((fc+1,1,2,64),dtype=torch.bfloat16,device=dev).normal_(std=.35,generator=rng)
    v=torch.empty_like(k).normal_(std=.35,generator=rng)
    names=["baseline",*args.arms,"control"]
    out=dict(scope="Single SM86 native sgl_kernel BF16 paged FA3. One frozen metadata update plus 16 full and 46 sliding native attention calls, or six layers in smoke mode. Uses the actual scored parent page kernels, same page geometry, official per-rank head counts, per-layer distinct queries, one bounded shared pair of synthetic K/V banks, and original full-verify schedule precomputation. H200 FP8 core, DFlash, dense/MoE and eight-rank communication are outside the measurement.",
        baseline_sha256=hashlib.sha256((root/"release_2026-11-18/candidate.diff").read_bytes()).hexdigest(),device=torch.cuda.get_device_name(),
        wrapper_sha256=wrapper_hash,seed=args.seed,fully_materialized_layers=len(layers),candidates=args.arms,
        schedule_mode="Separate fresh native plan for authoritative full/SWA geometry. The sliding plan is produced once inside the graph and the same graph-owned allocation is read by every matching layer. Each native attention may mutate and reset its own semaphore.",
        active=True,correct=True,complete=False,cases={})
    file=here/(("native_smoke_" if args.smoke else "native_stack_")+str(args.seed)+".json")
    def save():file.write_text(json.dumps(out,indent=2))
    save()
    for case in args.cases:
        b,length=map(int,case.split("x"))
        maxima=[length] if b==1 else [int(length*(.4+.6*i/(b-1))) for i in range(b)]
        assert sum(maxima)+b*n<fc
        table=torch.zeros((b+1,cap),device=dev,dtype=torch.int32)
        slots=torch.arange(1,fc+1,device=dev,dtype=torch.int32)
        total=fc//512*512;perm=torch.randperm(total//512,device=dev,generator=rng)
        slots[:total]=slots[:total].view(-1,512)[perm].flatten()
        off=0
        for i,t in enumerate(maxima):
            table[i+1,:t+n]=slots[off:off+t+n];off+=t+n
        limits=torch.tensor(maxima,device=dev,dtype=torch.int32)
        seq=torch.empty_like(limits);req=torch.empty(b,dtype=torch.int64,device=dev)
        mapping=torch.zeros(fc+2,dtype=torch.int64,device=dev)
        cuq=torch.arange(b+1,device=dev,dtype=torch.int32)*n
        queries=torch.empty((len(layers),b*n,16,64),device=dev,dtype=torch.bfloat16)
        def refresh(i):
            req.copy_(1+torch.randperm(b,device=dev,generator=rng))
            seq.copy_((limits[req-1]*(.65+.35*torch.rand(b,device=dev,generator=rng))).int())
            j=(seq+n-1)[:,None]-torch.arange(left+n,device=dev)[None,:]
            ids=table[req[:,None],j.clamp_min(0).long()][j>=0].long()
            mapping.zero_();mapping[ids]=(torch.arange(ids.numel(),device=dev)+i*773)%sc+1;mapping[-1]=-1
            queries.normal_(std=.31,generator=rng)
        refresh(0)
        def allocate():
            return dict(l=seq.clone(),cu=cuq.clone(),pt=torch.zeros((b,cap),device=dev,dtype=torch.int32),
                        sw=torch.zeros((b,cap),device=dev,dtype=torch.int32),idx=seq.clone(),
                        out=torch.empty_like(queries))
        states={name:allocate() for name in names}
        def plan(ll,window):
            return fa.get_scheduler_metadata(batch_size=b,max_seqlen_q=n,max_seqlen_k=cap,
                num_heads=16,num_heads_k=2,headdim=64,headdim_v=64,cache_seqlens=ll,qkv_dtype=torch.bfloat16,
                cu_seqlens_q=cuq,page_size=1,causal=True,window_size=window,num_splits=0)
        def run(name):
            st=states[name];l,cu,pt,sw,idx=tuple(st[t] for t in ("l","cu","pt","sw","idx"))
            args0=(l,cu,pt,table,req,cap,seq,n,1,sw,mapping)
            direct=name.startswith("direct");priv=name.startswith("private");reuse="schedule" in name
            if direct:fast(*args0,swa_suffix_tokens=left+n,direct_full_indices=idx)
            elif priv:snapshot(*args0,swa_suffix_tokens=left+n,full_block=2048,warps=4)
            else:baseline(*args0)
            std=plan(l,(-1,-1))
            if "fullbuf" not in st:st["fullbuf"]=torch.empty_like(std)
            st["fullbuf"].copy_(std)
            slide=None
            if reuse:
                slide=plan(l,(left,0))
            for j,i in enumerate(layers):
                full=(61-i)%4==0
                fa.flash_attn_with_kvcache(queries[j],k,v,cache_seqlens=l,cache_batch_idx=(idx if full and direct else None),
                    page_table=((table if direct else pt) if full else sw),cu_seqlens_q=cuq,cu_seqlens_k_new=cu,
                    max_seqlen_q=n,causal=True,window_size=(-1,-1) if full else (left,0),num_splits=0,
                    scheduler_metadata=st["fullbuf"] if full else slide,out=st["out"][j],ver=3)
        graphs={}
        for name in names:
            run(name)
            with torch.cuda.graph(graph:=torch.cuda.CUDAGraph()):run(name)
            graphs[name]=graph
        times={t:[] for t in names}
        rec=dict(times=times,matched_per_arm=0,arms={});out["cases"][case]=rec
        for i in range(4 if args.smoke else 8):
            refresh(i+1)
            rot=(i//2)%len(args.arms);in_order=args.arms[rot:]+args.arms[:rot]
            order=["baseline",*in_order,"control"]
            if i%2:order.reverse()
            for name in order:
                start=torch.cuda.Event(enable_timing=True);stop=torch.cuda.Event(enable_timing=True)
                start.record();graphs[name].replay();stop.record();stop.synchronize()
                times[name].append(start.elapsed_time(stop)*1000)
            ref=states["baseline"]["out"]
            assert torch.isfinite(ref).all()
            for name in names[1:]:
                alt=states[name]["out"]
                wrong=(ref.view(torch.int16)!=alt.view(torch.int16)).flatten(1).sum(1)
                if bool(wrong.any()):
                    rec["arms"][name]=dict(first_failed_replay=i+1,by_layer=wrong.tolist(),
                        max_abs=float((ref.float()-alt.float()).abs().max()))
                    out.update(correct=False);save()
                    print("native disagreement",case,name,rec["arms"][name],flush=True)
                    return
            rec["matched_per_arm"]+=ref.numel()
        print(case,{x:round(statistics.median(y),1) for x,y in times.items()},"outputs",rec["matched_per_arm"],flush=True);save()
        del states,graphs,graph,ref,alt
    out["complete"]=True;save()
if __name__=="__main__":main()
