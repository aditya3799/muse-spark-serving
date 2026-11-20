"""Falsify bounded runtime conversion on one concentrated local expert against the actual scored FFN."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import statistics
import sys
import torch
from slab import materialize,make_table

here=Path(__file__).resolve().parent
root=here.parents[1]
sys.path.insert(0,str(root/"research_2026-11-18_target_moe"))
import bench_experts as fixture
for p in (fixture.OLD/"humming_vendor",fixture.OLD/"phase2",root/"optimization_marlin_tiles_2026-11-15"):
    sys.path.insert(0,str(p))
from kernel_runtime import bootstrap
bootstrap(root/"sglang")
from fixture_tuning_cache import cached_threshold_for_preparation
from sglang.kernels.ops.activation.activation import silu_and_mul
ops=fixture.SOURCE/"kernels/ops/moe"
scale=fixture.module("sglang.kernels.ops.moe.muse_spark_humming_scale",ops/"muse_spark_humming_scale.py")
fixture.module("sglang.kernels.ops.moe.muse_spark_humming_split",ops/"muse_spark_humming_split.py")
source=fixture.SOURCE/"srt/layers/muse_spark/muse_spark_nvfp4_moe.py"
names={"_prepare_humming_experts","_humming_expert_outputs","_humming_expert_outputs_default"}
ns={"torch":torch}; tree=ast.parse(source.read_text())
exec(compile(ast.Module([n for n in tree.body if getattr(n,"name",None) in names],[]),str(source),"exec"),ns)

def main():
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=81239)
    p.add_argument("--rows",nargs="+",type=int,default=[128,512,1024])
    p.add_argument("--diagnose",action="store_true")
    args=p.parse_args()
    torch.manual_seed(args.seed);e,h=32,4096
    base=hashlib.sha256((root/"release_2026-11-18/candidate.diff").read_bytes()).hexdigest()
    assert base=="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
    report=dict(baseline_sha256=base,scope="One active local expert among 32, at slot zero; all other slots are remote. Both paths consume the same rank input, packed coefficients and active output destinations. This is an optimistic concentrated-expert screen before general routing or Hopper grouped GEMM. Dense time includes decoding the sole resident Humming weight format twice per forward.",seed=args.seed,rows=[],shape_checks={},correct=True,complete=False)
    path=here/f"slab_gpu_{args.seed}.json"
    if args.diagnose:path=here/f"slab_stage_{args.seed}.json"
    def save():path.write_text(json.dumps(report,indent=2))
    save()
    raw=torch.nn.Module(); samples={}
    for stage,n in (("w13",2*h),("w2",h)):
        w=torch.randint(256,(e,n,h//2),device="cuda",dtype=torch.uint8)
        sf=torch.randint(33,75,(e,n,h//16),device="cuda",dtype=torch.uint8)
        ii=torch.randint(n,(16000,),device="cuda");jj=torch.randint(h,(16000,),device="cuda")
        codes=(w[0,ii,jj//2].int()>>(4*(jj%2)))&15
        samples[stage]=(ii,jj,sf[0,ii,jj//16].long(),codes.long())
        for key,t in (("weight",w),("weight_scale",sf.view(torch.float8_e4m3fn)),("weight_scale_2",torch.linspace(.125,.5,e,device="cuda").bfloat16())):
            setattr(raw,stage+"_"+key,torch.nn.Parameter(t,requires_grad=False))
    with cached_threshold_for_preparation(here):layer=ns["_prepare_humming_experts"](raw,torch)
    del raw,w,sf,t
    slab=torch.empty((1,2*h,h),device="cuda",dtype=torch.bfloat16)
    expert=torch.zeros(1,device="cuda",dtype=torch.int32)
    table=make_table("cuda")
    stages={}
    for name,n in (("w13",2*h),("w2",h)):
        meta=layer.humming_metas[name]
        assert not meta.should_apply_bs_on_c and not meta.use_fused_e8m0_scale and meta.weight_scale_group_size==16
        wg="wgmma" in str(meta.mma_type).lower()
        stages[name]=(getattr(layer,meta.weight_name),getattr(layer,meta.weight_scale_name),wg)
        w,sf,wg=stages[name];dst=slab[:,:n]
        materialize(w,sf,table,expert,dst,wg)
        ii,jj,ss,cc=samples[name]
        actual=dst[0,ii,jj].view(torch.int16);gold=table[ss,cc].view(torch.int16)
        bad=int((actual!=gold).sum())
        report["shape_checks"][name]=dict(shape=[n,h],sampled=ii.numel(),bits_wrong=bad,mma=str(meta.mma_type))
        save();assert bad==0,(name,bad)
    def decode(name):
        w,sf,wg=stages[name];n=2*h if name=="w13" else h
        return materialize(w,sf,table,expert,slab[:,:n],wg)[0]
    for m in args.rows:
        x=torch.randn((m,h),device="cuda",dtype=torch.bfloat16)*.125
        ids=torch.full((m,8),-1,device="cuda",dtype=torch.int32);ids[:,0]=0
        one=torch.zeros((m,1),device="cuda",dtype=torch.int32)
        spatial=torch.empty((m,8,h),device="cuda",dtype=torch.bfloat16)
        def parent():
            out,_=ns["_humming_expert_outputs"](layer,x,ids,torch,defer_downscale=True,global_experts=256)
            return out[:,0]
        if args.diagnose:
            from stage_compare import examine
            report["rows"].append(dict(rows=m,stages=examine(parent,decode,m)))
            save();print(m,report["rows"][-1],flush=True)
            continue
        def candidate():
            gu=torch.mm(x,decode("w13").T).mul_(4)
            scale.scale_active_rows(gu,one,layer.w13_post_scale)
            hidden=torch.empty_like(x)
            silu_and_mul(gu,hidden,expert_ids=one.flatten(),expert_step=1)
            down=torch.mm(hidden,decode("w2").T).mul_(4)
            spatial[:,0].copy_(down)
            return spatial[:,0]
        cases=dict(rows=m,differences=[],parent_us=[],candidate_us=[],control_us=[])
        report["rows"].append(cases)
        for i in range(5):
            x.normal_(std=.16 if i%2 else .03)
            a=parent().clone();b=candidate()
            mask=a.view(torch.int16)!=b.view(torch.int16)
            cases["differences"].append(dict(count=int(mask.sum()),values=a.numel(),max_abs=float((a.float()-b.float()).abs().max())))
            if mask.any():
                report.update(correct=False,reward=-100.,verdict="INCORRECT",complete=True)
                save();print(m,cases["differences"][-1],"reject exact dense route",flush=True);return
        for _ in range(2):parent();candidate()
        for i in range(10):
            x.normal_()
            order=[("parent",parent),("candidate",candidate),("control",parent)]
            for label,fn in (order if i%2==0 else reversed(order)):
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record();fn();b.record();b.synchronize();cases[label+"_us"].append(a.elapsed_time(b)*1000)
        save();print(m,statistics.median(cases["parent_us"])/statistics.median(cases["candidate_us"]),flush=True)
    report["complete"]=True;save()
if __name__=="__main__":main()
