"""Run actual scored and candidate Humming expert helpers at the same routed inputs."""
import argparse
import ast
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

here=Path(__file__).resolve().parent
root=here.parents[1]
sys.path.insert(0,str(root/"research_2026-11-18_target_moe"))
import bench_experts as fixture
for path in (fixture.OLD/"humming_vendor",fixture.OLD/"phase2",root/"optimization_marlin_tiles_2026-11-15"):
    sys.path.insert(0,str(path))
from kernel_runtime import bootstrap
bootstrap(root/"sglang")
sys.path.insert(0,str(root/"research_2026-11-18_rl_prefill/gemm"))
from fixture_tuning_cache import cached_threshold_for_preparation
import torch
import scale_order

def extract(path):
    names={"_prepare_humming_experts","_humming_expert_outputs_default","_humming_expert_outputs"}
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns={"torch":torch}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),ns)
    return ns

parser=argparse.ArgumentParser()
parser.add_argument("--seed",type=int,required=True)
parser.add_argument("--out",required=True,type=Path)
parser.add_argument("--rows",type=int,nargs="+",default=[753,3012])
parser.add_argument("--pairs",type=int,default=6)
parser.add_argument("--stage-check", action="store_true")
args=parser.parse_args()
assert args.pairs>=4 and args.pairs%2==0
scored_sha=hashlib.sha256((fixture.RELEASE/"candidate.diff").read_bytes()).hexdigest()
assert scored_sha=="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
report=dict(agent="experts",candidate="scale_register_order",scope="W13+scored scaling/SiLU+W2+scored RMS/weighted sum",
           baseline_sha256=scored_sha,candidate_sha256=hashlib.sha256((here/"candidate.diff").read_bytes()).hexdigest(),
           seed=args.seed,active=False,correct=False,complete=False,unit="us",gpu=torch.cuda.get_device_name(),
           bench={"instances":1,"shapes":{}})
def save():args.out.write_text(json.dumps(report,indent=2)+"\n")
save()
moe=fixture.SOURCE/"kernels/ops/moe"
fixture.module("sglang.kernels.ops.moe.muse_spark_humming_scale",moe/"muse_spark_humming_scale.py")
key="sglang.kernels.ops.moe.muse_spark_humming_split"
sk=fixture.module(key,moe/"muse_spark_humming_split.py")
ck=fixture.module("candidate_index_split",here/"muse_spark_humming_split.py")
finish=fixture.module("scale_order_scored_finalizer",moe/"muse_spark_post_expert_finalize_triton.py").fused_post_expert_norm_weighted_sum
sys.modules["sglang.kernels.ops.moe.scale_order"]=scale_order
sn=extract(fixture.SOURCE/"srt/layers/muse_spark/muse_spark_nvfp4_moe.py")
cn=extract(here/"muse_spark_nvfp4_moe.py")
raw=torch.nn.Module();h=4096;e=32
torch.manual_seed(args.seed)
for stage,n in (("w13",2*h),("w2",h)):
    for name,v in (("weight",torch.randint(0,256,(e,n,h//2),device="cuda",dtype=torch.uint8)),
                   ("weight_scale",torch.randint(40,68,(e,n,h//16),device="cuda",dtype=torch.uint8).view(torch.float8_e4m3fn)),
                   ("weight_scale_2",torch.linspace(.125,.5,e,device="cuda",dtype=torch.bfloat16))):
        setattr(raw,stage+"_"+name,torch.nn.Parameter(v,requires_grad=False))
# Stamp every finite signed E4M3 scale on each expert. Both timed paths use these buffers.
finite_codes = torch.tensor([code for code in range(256) if code not in (127,255)],device="cuda",dtype=torch.uint8)
for stage in ("w13","w2"):
    getattr(raw,stage+"_weight_scale").view(torch.uint8)[:,:254,0].copy_(finite_codes)
report["finite_e4m3_scale_bytes_planted_per_expert"] = int(finite_codes.numel())
with cached_threshold_for_preparation(here):
    layer=sn["_prepare_humming_experts"](raw,torch)
del raw,v
used=set()
stage_inputs={}
from scale_evidence import observe
observe(scale_order,report)
original=scale_order.scale_order_forward
def audited(layer,**kwargs):
    if args.stage_check and recording[0]:
        stage_inputs.setdefault(kwargs["sublayer_name"],[]).append(dict(kwargs))
    before=scale_order.kernel_handle.cache_info()
    result=original(layer,**kwargs)
    after=scale_order.kernel_handle.cache_info()
    if before.hits+before.misses<after.hits+after.misses:
        used.add((kwargs["valid_shape_m"],kwargs["sublayer_name"]))
    return result
scale_order.scale_order_forward=audited
norm=torch.linspace(.7,1.2,h,device="cuda",dtype=torch.bfloat16)
recording=[False]
for m in args.rows:
    x=torch.empty((m,h),dtype=torch.bfloat16,device="cuda")
    ids=torch.empty((m,8),dtype=torch.int32,device="cuda")
    scores=torch.empty((m,8),device="cuda")
    def update(seed,cluster):
        gen=torch.Generator().manual_seed(seed)
        weights=torch.rand(m,256,generator=gen)
        if cluster:weights[::4,:8]+=2
        routes=weights.topk(8).indices
        ids.copy_(torch.where(routes<32,routes,-1))
        torch.manual_seed(seed)
        x.normal_(std=.023);scores.uniform_();scores.div_(scores.sum(1,keepdim=True))
    def run(changed):
        sys.modules[key]=ck if changed else sk
        helpers=cn if changed else sn
        y,loc=helpers["_humming_expert_outputs"](layer,x,ids,torch,defer_downscale=True,global_experts=256)
        return finish(y,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=loc,expert_output_scale=layer.w2_post_scale)
    update(args.seed+1,False)
    if args.stage_check:
        from stage_pair import check
        stage_inputs.clear();recording[0]=True
        run(True);torch.cuda.synchronize();recording[0]=False
        report.setdefault("stage_trials",{})[str(m)]=check(m,layer,stage_inputs,args.seed,args.pairs,scale_order.scale_order_forward)
        stage_inputs.clear();save()
    graphs={};outputs={}
    for name in ("baseline","candidate","control"):
        changed=name=="candidate"
        run(changed);torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outputs[name]=run(changed)
        graphs[name]=g
    assert all((m*8,stage) in used for stage in ("w13","w2")),used
    def timed(name):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();graphs[name].replay();end.record();end.synchronize()
        return start.elapsed_time(end)*1000
    for profile in ("uniform","clustered"):
        name=f"M{m}_{profile}"
        r={"weight":1,"times":{k:[] for k in graphs},"parity":[],"order":[]}
        report["bench"]["shapes"][name]=r
        for i in range(args.pairs):
            update(args.seed+19+i*31,profile=="clustered")
            names=(["baseline","candidate","control"] if i%2==0 else ["control","candidate","baseline"])
            for label in names:r["times"][label].append(timed(label))
            ok=bool(torch.isfinite(outputs["candidate"]).all()) and torch.equal(outputs["candidate"],outputs["baseline"])
            r["parity"].append(ok);r["order"].append(names);save()
            assert ok,(m,i)
        print(name,statistics.median(r["times"]["baseline"])/statistics.median(r["times"]["candidate"]),flush=True)
    del graphs,outputs,g
    torch.cuda.empty_cache()
report.update(active=True,correct=True,complete=True,stage_calls=sorted(used));save()



