"""Local released-slot qualification for exclusive weights feeding the unchanged native Humming FFN."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch
from exclusive_pack import NativeFallback,install_layer

here=Path(__file__).resolve().parent
root=here.parents[1]
sys.path.insert(0,str(root/"research_2026-11-18_rl/expert_slab"))


def main():
    p=argparse.ArgumentParser();p.add_argument("--seed",required=True,type=int)
    p.add_argument("--rows",nargs="+",type=int,default=[8,32,128])
    p.add_argument("--rounds",type=int,default=4);p.add_argument("--direct",action="store_true");p.add_argument("--vector",action="store_true");p.add_argument("--scales-only",action="store_true");args=p.parse_args()
    import gpu_probe as parent
    finish=parent.fixture.module("exclusive_finalize",parent.ops/"muse_spark_post_expert_finalize_triton.py").fused_post_expert_norm_weighted_sum
    digest=hashlib.sha256((root/"release_2026-11-18/candidate.diff").read_bytes()).hexdigest()
    assert digest=="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
    e,h=32,4096;torch.manual_seed(args.seed)
    path=here/(("scales_native_" if args.scales_only else "vector_native_" if args.vector else "direct_native_" if args.direct else "fallback_native_")+str(args.seed)+".json")
    report=dict(seed=args.seed,device=torch.cuda.get_device_name(),parent=digest,rows=[],complete=False,correct=None)
    def save():path.write_text(json.dumps(report,indent=2))
    save()
    direct_fn=None
    if args.direct or args.vector or args.scales_only:
        from exclusive_direct import bind_frozen,CanonicalMethod
        CanonicalMethod.vector=args.vector
        direct_fn=bind_frozen(parent.ns["_humming_expert_outputs_default"],sys.modules["sglang.kernels.ops.moe.muse_spark_humming_split"])
    raw=torch.nn.Module()
    for stage,n in (("w13",2*h),("w2",h)):
        for key,t in (("weight",torch.randint(256,(e,n,h//2),device="cuda",dtype=torch.uint8)),
                      ("weight_scale",torch.randint(42,70,(e,n,h//16),device="cuda",dtype=torch.uint8).view(torch.float8_e4m3fn)),
                      ("weight_scale_2",torch.linspace(.125,.5,e,device="cuda").bfloat16())):
            setattr(raw,stage+"_"+key,torch.nn.Parameter(t,requires_grad=False))
    with parent.cached_threshold_for_preparation(here):
        layer=parent.ns["_prepare_humming_experts"](raw,torch)
    del raw,t
    # A duplicate weights-only parameter registry exists solely for trustworthy A/B and is measured outside timing.
    canonical=copy.copy(layer);canonical._parameters=layer._parameters.copy()
    if args.scales_only:
        from shared_scales import install_scales
        install_scales(canonical)
    else:
        install_layer(canonical)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    norm=torch.linspace(.7,1.2,h,device="cuda").bfloat16()
    with torch.cuda.stream(stream):
        arena=NativeFallback(e*2*h*h*(1 if args.scales_only else 9)//16,e,torch.device("cuda"))
        for m in args.rows:
            x=torch.empty((m,h),device="cuda",dtype=torch.bfloat16);ids=torch.empty((m,8),device="cuda",dtype=torch.int32)
            scores=torch.empty((m,8),device="cuda")
            def inputs(j,mode):
                g=torch.Generator().manual_seed(args.seed+17*m+j)
                keys=torch.rand(m,256,generator=g)
                if mode=="skew":keys[:,0]+=2;keys[:,32:39]+=2
                k=keys.topk(8).indices
                ids.copy_(torch.where(k<32,k,-1))
                x.copy_(torch.randn(m,h,generator=g)*(.05 if j%2 else .13))
                s=torch.rand(m,8,generator=g);scores.copy_(s/s.sum(-1,keepdim=True))
            def fn(arm):
                layer_view=arena.view(canonical,ids,lazy=bool(direct_fn)) if arm=="candidate" else layer
                run=direct_fn if arm=="candidate" and direct_fn else parent.ns["_humming_expert_outputs"]
                y,loc=run(layer_view,x,ids,torch,defer_downscale=True,global_experts=256)
                z=finish(y,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=loc,
                         expert_output_scale=layer.w2_post_scale)
                return y,z
            inputs(0,"skew")
            graphs={};outs={}
            for arm in ("parent","candidate"):
                fn(arm);stream.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):
                    outs[arm]=fn(arm)
                graphs[arm]=graph
            for mode in ("skew","uniform"):
                rec=dict(m=m,route=mode,live=[],raw_bit_mismatches=[],final_bit_mismatches=[],times_us={k:[] for k in ("parent","candidate","control")})
                report["rows"].append(rec)
                for j in range(args.rounds):
                    inputs(j,mode)
                    for arm in ("parent","candidate"):graphs[arm].replay()
                    live=ids>=0;rec['live'].append(len(ids[live].unique()))
                    yy,zz=outs['candidate'];orig,base=outs['parent']
                    rec['raw_bit_mismatches'].append(int((yy[live].view(torch.int16)!=orig[live].view(torch.int16)).sum()))
                    rec['final_bit_mismatches'].append(int((zz.view(torch.int16)!=base.view(torch.int16)).sum()))
                    if rec['raw_bit_mismatches'][-1] or rec['final_bit_mismatches'][-1]:
                        report.update(correct=False,complete=True);save();print("rejected",m,mode,flush=True);return
                    order=("candidate","parent","control") if j%2 else ("parent","candidate","control")
                    for arm in order:
                        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                        a.record();graphs['parent' if arm=='control' else arm].replay();b.record();b.synchronize()
                        rec['times_us'][arm].append(a.elapsed_time(b)*1000)
                if direct_fn:report["python_dispatch_counts"]=CanonicalMethod.calls.copy()
                print(m,mode,rec['live'],flush=True);save()
            stream.synchronize();del outs,graphs,graph
    report.update(correct=True,complete=True);save()
if __name__=="__main__":main()
