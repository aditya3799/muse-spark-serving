"""Four serial, state-coupled Muse blocks with native paged attention and scored Humming."""
import argparse
import hashlib
import inspect
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace as S
import torch
import torch.nn.functional as F
import sgl_kernel.flash_attn as fa
import operators as op

H,ROOT=op.H,op.ROOT
sys.path.insert(0,str(H.parent/"decode_pages/global_rows"))
from real_layout import load
origpages,directpages=load("parent_metadata"),load("metadata")

def main():
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int,required=True)
    p.add_argument("--cases",nargs="+",default=["1x8192","4x8192","16x4096"])
    p.add_argument("--rounds",type=int,default=8);p.add_argument("--page",choices=["parent","direct"],default="parent")
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    assert args.rounds%2==0
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    gpu=torch.cuda.get_device_name()
    report=dict(agent="prefill",candidate="packet+swa_schedule+grouped_at_supported_M128",
        scope="Four consecutive TP8-shaped decoder blocks; real native BF16 paged FA3 consumes expanded E4M3, original 32-expert rank-local Humming, source projections/RMS/residual/router and attn/post-FFN feedback. Synthetic weights and KV history, one shared expert bank (four distinct dense weight sets), no remote sums or peer traffic.",
        gpu=gpu,seed=args.seed,baseline_sha256=sha(ROOT/"release_2026-11-18/candidate.diff"),correct=True,active=True,
        complete=False,started=time.time(),cases={},bench=dict(instances=1,shapes={}),
        source=dict(fa=sha(inspect.getfile(fa)),humming=sha(op.hg.source),qkv=sha(op.QKDIR/"candidate.diff"),
                    scheduler=sha(H.parent/"swa_scheduler/candidate.diff"),grouped=sha(inspect.getfile(op.group))),
        notes=["One shared rank-local packed W13/W2 bank (32 of 256 experts) and 4 independent full+sliding cache sets.",
               "FP8 Q/K/V on both arms expanded identically into BF16 for installed SM86 native attention. Only new K/V rows use a GPU bridge per layer.",
               "TP sums after O and after local expert finalize take the single rank partial. Missing pooled/unpooled columns precomputed from the identical live reference inputs per replay. Timed work recomputes rank 0's own 1/8 column shard.",
               "M>8 uses scored Hopper split2 router arithmetic on local GPU. Small FP32 cuBLAS, scored dual-stream projection/routing and native top8. All timing brackets include 4 actual FFNs including W13/W2 at full inner width.",
               "Windowed layers use 2047-left local and rotary; full layer uses no rotary. Model scale 5.4730064863875/sqrt(64). Graphs first refresh old metadata and full-causal scheduler; candidate additionally shares a separate local plan."])
    assert report["source"]["fa"]=="610747f2c68495c5f3ba67f1247f0a1b4e4f7df116e2f89cd8e8983f0271f9ac"
    def save():
        tmp=args.out.with_suffix(".tmp");tmp.write_text(json.dumps(report,indent=2));tmp.replace(args.out)
    save()
    torch.manual_seed(args.seed);torch.set_float32_matmul_precision("high")
    print("preparing real rank bank",flush=True);bank=op.bank()
    torch.cuda.empty_cache()
    print("real Humming bank ready",torch.cuda.memory_allocated(),flush=True)
    rng=torch.Generator(device="cuda").manual_seed(args.seed+32)
    lut=torch.arange(65536,device="cuda",dtype=torch.int32).short().view(op.b16).to(op.f8).view(torch.uint8)
    cap=131072;left=2047;n=8;dev="cuda"
    inv=500000.**(-torch.arange(0,64,2,device=dev).float()/64)
    phase=torch.arange(cap+1,device=dev).float()[:,None]*inv
    rotary=torch.cat([phase.cos(),phase.sin()],dim=1).contiguous();del phase
    for case in args.cases:
        b,l=map(int,case.split("x"));m=b*n;pages=b*(l+n)+1
        torch.cuda.reset_peak_memory_stats()
        layers=[op.Block(m,pages,j) for j in range(4)]
        side=torch.cuda.Stream()
        source=torch.zeros((b+1,cap),dtype=torch.int32,device=dev)
        shuff=torch.randperm(pages-1,device=dev,generator=rng,dtype=torch.int32).view(b,l+n)+1
        source[1:,:l+n].copy_(shuff);mapping=torch.empty(pages+1,device=dev,dtype=torch.int64)
        mapping[:pages]=torch.arange(pages,device=dev);mapping[1:pages]=torch.randperm(pages-1,device=dev,generator=rng)+1;mapping[-1]=-1
        req=torch.arange(1,b+1,dtype=torch.int64,device=dev)
        seq=torch.zeros(b,device=dev,dtype=torch.int32);pos=torch.empty(m,device=dev,dtype=torch.int64)
        raw=torch.empty(m,8192,device=dev,dtype=torch.float32)
        loc=torch.empty(m,device=dev,dtype=torch.int64);swa_loc=torch.empty_like(loc)
        cuq=torch.arange(b+1,device=dev,dtype=torch.int32)*n
        def update(i):
            raw.normal_(generator=rng);req.copy_(1+torch.randperm(b,device=dev,generator=rng))
            seq.copy_(torch.randint(l//2,l+1,(b,),device=dev,dtype=torch.int32,generator=rng))
            pos.copy_((seq[:,None]+torch.arange(n,device=dev)).flatten())
            loc.copy_(source[req[:,None],pos.view(b,n)].flatten());swa_loc.copy_(mapping[loc])
            for block in layers:
                block.bias.zero_()
                if i%2:block.bias[:8].fill_(.08)
        def poison():
            for block in layers:
                slots=loc if block.full else swa_loc
                for buf in [block.k8,block.v8]:buf.view(torch.uint8).index_fill_(0,slots,127)
                for buf in [block.k,block.v]:buf.index_fill_(0,slots,float("nan"))
        update(0)
        names=["baseline","candidate","control"]
        states={}
        for a in names:
            states[a]=S(l=seq.clone(),cu=cuq.clone(),pt=torch.zeros_like(source[:b]),
                sw=torch.zeros_like(source[:b]),idx=seq.clone(),full=None)
        def plan(st,window):
            return fa.get_scheduler_metadata(batch_size=b,max_seqlen_q=n,max_seqlen_k=cap,num_heads=16,num_heads_k=2,
                headdim=64,headdim_v=64,cache_seqlens=st.l,qkv_dtype=op.b16,cu_seqlens_q=cuq,
                cu_seqlens_k_new=None,page_size=1,causal=True,window_size=window,num_splits=0)
        def body(a,stage=False):
            cand=a=="candidate";direct=cand and args.page=="direct";st=states[a]
            pageargs=(st.l,st.cu,st.pt,source,req,cap,seq,n,1,st.sw,mapping)
            if direct:directpages(*pageargs,swa_suffix_tokens=left+n,direct_full_indices=st.idx)
            else:origpages(*pageargs)
            std=plan(st,(-1,-1))
            if st.full is None:st.full=torch.empty_like(std)
            st.full.copy_(std)
            slide=plan(st,(left,0)) if cand else None
            stream=raw;hidden=F.rms_norm(stream,(8192,),eps=1e-5).to(op.b16)
            outputs=[]
            for block in layers:
                slots=loc if block.full else swa_loc
                q,gate=block.entry(hidden,pos,rotary,slots,lut,cand)
                att=fa.flash_attn_with_kvcache(q,block.k,block.v,cache_seqlens=st.l,
                    cache_batch_idx=st.idx if block.full and direct else None,page_table=(source if direct else st.pt) if block.full else st.sw,
                    cu_seqlens_q=cuq,cu_seqlens_k_new=st.cu,max_seqlen_q=n,causal=True,
                    window_size=(-1,-1) if block.full else (left,0),softmax_scale=5.4730064863875/8,
                    scheduler_metadata=st.full if block.full else slide,num_splits=0,ver=3)
                hidden,stream,ids=block.finish(att,gate,stream,bank,side,stage,cand and m==128)
                outputs.append((att,hidden,stream,ids))
            return outputs
        print("case",case,"warmup",flush=True)
        body("baseline",True)
        graphs={};outs={}
        for a in names:
            body(a);torch.cuda.synchronize()
            with torch.cuda.graph(g:=torch.cuda.CUDAGraph()):
                outs[a]=body(a)
            graphs[a]=g
        counts=[0,0,0,0];changes=0;previous=None;local_counts=[]
        times={a:[] for a in names};cs=dict(cases=case,live_cache_pages_per_layer=pages,order=[],local_route_slots=local_counts)
        report["cases"][case]=cs
        report["bench"]["shapes"][case]=dict(weight=1,times=times)
        print("case",case,"all graphs ready",flush=True)
        for i in range(args.rounds):
            update(i+1);poison();body("baseline",True);torch.cuda.synchronize()
            order=names if i%2==0 else names[::-1];cs["order"].append(list(order))
            for a in order:
                poison()
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record();graphs[a].replay();end.record();end.synchronize();times[a].append(begin.elapsed_time(end)*1000)
            for j,r in enumerate(outs["baseline"]):
                for stage,ref in enumerate(r):
                    for a in ["candidate","control"]:
                        v=outs[a][j][stage]
                        if not torch.equal(v,ref) or not bool(torch.isfinite(v).all()):
                            report["correct"]=False
                            cs["failure"]=dict(round=i,block=j,stage=["attention","next_attention_input","fp32_residual","global_expert_ids"][stage],arm=a,
                                wrong=int((v!=ref).sum()),max_abs=float((v.float()-ref.float()).abs().max()))
                            save();print("failed",cs["failure"],flush=True);return
                    counts[stage]+=ref.numel()
            ids=torch.stack([r[3] for r in outs["baseline"]]);local_counts.append(int((ids<32).sum()))
            if previous is not None:changes+=int((ids!=previous).sum())
            previous=ids.clone()
            cs.update(bitwise_values_per_stage=counts,changed_route_slots=changes,peak_allocated=int(torch.cuda.max_memory_allocated()),
                      peak_reserved=int(torch.cuda.max_memory_reserved()),median_us={a:statistics.median(t) for a,t in times.items()})
            save()
        assert changes>0 and min(local_counts)>0
        cs["speedup"]=cs["median_us"]["baseline"]/cs["median_us"]["candidate"];save();print(case,cs["median_us"],cs["speedup"],flush=True)
        del outs,graphs,g,layers,r,v,ref
        torch.cuda.synchronize();torch.cuda.empty_cache()
    report["complete"]=True;save()
if __name__=="__main__":main()
