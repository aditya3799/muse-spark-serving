"""Manual, single-Hopper hardware gate against the frozen scored parent. No remote actions."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch

here=Path(__file__).resolve().parent
root=here.parents[1]
sys.path.insert(0,str(root/"research_2026-11-18_rl/expert_slab"))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",required=True,type=int)
    p.add_argument("--rows",nargs="+",type=int,default=[128,768,3072])
    p.add_argument("--rounds",type=int,default=8)
    p.add_argument("--transient",action="store_true",help="include per-step group weight repacking")
    p.add_argument("--shared-weights",action="store_true",help="point directly at the existing Humming weight tensor")
    args=p.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(9,0):
        raise SystemExit("Requires a local SM90a GPU and the pinned SGLang/Humming environment")
    import gpu_probe as parent
    from group_runtime import GroupedFFN
    finish=parent.fixture.module("cutlass_gate_finalize",parent.ops/"muse_spark_post_expert_finalize_triton.py").fused_post_expert_norm_weighted_sum
    signature=hashlib.sha256((root/"release_2026-11-18/candidate.diff").read_bytes()).hexdigest()
    assert signature=="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
    e,h=32,4096
    report=dict(seed=args.seed,device=torch.cuda.get_device_name(),parent_sha256=signature,
                native_sha256=hashlib.sha256((here/("libgroup_shared_sm90a.so" if args.shared_weights else "libgroup_g16_sm90a.so")).read_bytes()).hexdigest(),
                correct=None,complete=False,exclusive=not args.transient,shared_weights=args.shared_weights,cases=[])
    path=here/f"hopper_qualification_{args.seed}.json"
    def save(): path.write_text(json.dumps(report,indent=2))
    save()
    torch.manual_seed(args.seed)
    raw=torch.nn.Module()
    for stage,n in (("w13",2*h),("w2",h)):
        for name,value in (("weight",torch.randint(256,(e,n,h//2),device="cuda",dtype=torch.uint8)),
                           ("weight_scale",torch.randint(32,77,(e,n,h//16),device="cuda",dtype=torch.uint8).view(torch.float8_e4m3fn)),
                           ("weight_scale_2",torch.linspace(.125,.5,e,device="cuda").bfloat16())):
            setattr(raw,stage+"_"+name,torch.nn.Parameter(value,requires_grad=False))
    with parent.cached_threshold_for_preparation(here):
        layer=parent.ns["_prepare_humming_experts"](raw,torch)
    del raw,value
    candidate_layer=layer
    if not args.transient:
        from exclusive_pack import install_layer
        candidate_layer=copy.copy(layer)
        candidate_layer._parameters=layer._parameters.copy()
        if args.shared_weights:
            from shared_scales import install_scales
            install_scales(candidate_layer)
        else:
            install_layer(candidate_layer)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        norm=torch.linspace(.71,1.31,h,device="cuda",dtype=torch.bfloat16)
        for m in args.rows:
            ctx=GroupedFFN(max(args.rows),h,h,e,torch.device("cuda",torch.cuda.current_device()),exclusive=not args.transient,shared_weights=args.shared_weights)
            try:
                x=torch.empty((m,h),device="cuda",dtype=torch.bfloat16)
                ids=torch.empty((m,8),device="cuda",dtype=torch.int32)
                score=torch.empty((m,8),device="cuda")
                def inputs(i,mode):
                    gen=torch.Generator().manual_seed(args.seed+m+i)
                    keys=torch.rand(m,256,generator=gen)
                    if mode=="skew":keys[::2,:8]+=2
                    local=keys.topk(8).indices
                    ids.copy_(torch.where(local<e,local,-1))
                    if mode=="remote":ids.fill_(-1)
                    x.copy_(torch.randn(m,h,generator=gen)*(.17 if i%2 else .023))
                    s=torch.rand(m,8,generator=gen)
                    score.copy_(s/s.sum(-1,keepdim=True))
                def call(arm):
                    if arm=="candidate":
                        y,loc=ctx.forward(candidate_layer,x,ids,defer_downscale=True)
                    else:
                        y,loc=parent.ns["_humming_expert_outputs"](layer,x,ids,torch,defer_downscale=True,global_experts=256)
                    z=finish(y,score,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=loc,
                             expert_output_scale=layer.w2_post_scale)
                    return y,z
                inputs(0,"uniform")
                graphs={}
                outputs={}
                for arm in ("parent","candidate"):
                    call(arm)
                    stream.synchronize()
                    graphs[arm]=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graphs[arm],stream=stream):
                        outputs[arm]=call(arm)
                for mode in ("uniform","skew","remote"):
                    row=dict(m=m,route=mode,parity=[],times_us={k:[] for k in ("parent","candidate","control")})
                    report["cases"].append(row)
                    for j in range(4):
                        inputs(j,mode)
                        for a in graphs.values():a.replay()
                        live=ids>=0
                        differing=[]
                        for ix in (0,1):
                            aa=outputs["parent"][ix];bb=outputs["candidate"][ix]
                            if ix==0:aa,bb=aa[live],bb[live]
                            differing.append(int((aa.view(torch.int16)!=bb.view(torch.int16)).sum()))
                        row["parity"].append(dict(raw_live_columns=differing[0],final=differing[1]))
                        if any(differing):
                            report.update(correct=False,complete=True)
                            save()
                            print("Rejected on bitwise mismatch",m,mode,j,differing)
                            return
                    for j in range(args.rounds):
                        inputs(17+j,mode)
                        order=("parent","candidate","control") if j%2==0 else ("candidate","parent","control")
                        for arm in order:
                            t0=torch.cuda.Event(enable_timing=True);t1=torch.cuda.Event(enable_timing=True)
                            t0.record();graphs["parent" if arm=="control" else arm].replay();t1.record();t1.synchronize()
                            row["times_us"][arm].append(t0.elapsed_time(t1)*1000)
                    save()
                stream.synchronize()
                del graphs,outputs
            finally:
                ctx.close()
    report.update(correct=True,complete=True)
    save()

if __name__=="__main__":main()
