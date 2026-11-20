"""Paired frozen-parent CUDA event ranges on genuine kernel-task route fixtures."""
import argparse
import ast
import collections
import contextlib
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

here=Path(__file__).resolve().parent
root=here.parents[1]
sys.path[:0]=[str(root/"research_2026-11-18_target_moe"),str(here.parent)]
import bench_experts as fixture
for path in (fixture.OLD/"humming_vendor",fixture.OLD/"phase2",root/"optimization_marlin_tiles_2026-11-15"):
    sys.path.insert(0,str(path))
from kernel_runtime import bootstrap
bootstrap(root/"sglang")
import torch
from dense_envelope import DenseEnvelope
from fixture_tuning_cache import cached_threshold_for_preparation
from humming.layer import HummingMethod
from sglang.srt.layers.moe import fused_moe_triton as align
from sglang.kernels.ops.activation import activation
sys.path.insert(0,str(root.parent/"muse_moe_w4a16"))
ref=fixture.module("official_expert_routing_for_profile",root.parent/"muse_moe_w4a16/reference.py")
moe=fixture.SOURCE/"kernels/ops/moe"
scale=fixture.module("sglang.kernels.ops.moe.muse_spark_humming_scale",moe/"muse_spark_humming_scale.py")
split=fixture.module("sglang.kernels.ops.moe.muse_spark_humming_split",moe/"muse_spark_humming_split.py")
finish=fixture.module("profile_scored_finish",moe/"muse_spark_post_expert_finalize_triton.py").fused_post_expert_norm_weighted_sum
path=fixture.SOURCE/"srt/layers/muse_spark/muse_spark_nvfp4_moe.py"
functions=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in
           ("_prepare_humming_experts","_humming_expert_outputs_default","_humming_expert_outputs")]
ns={"torch":torch};exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),"exec"),ns)
parser=argparse.ArgumentParser()
parser.add_argument("--out",type=Path,required=True);parser.add_argument("--seed",type=int,default=818731)
parser.add_argument("--rows",type=int,nargs="+",default=[128,768,16384])
args=parser.parse_args()
med=statistics.median
sha=hashlib.sha256((fixture.RELEASE/"candidate.diff").read_bytes()).hexdigest()
assert sha=="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
report=dict(parent_sha256=sha,seed=args.seed,gpu=torch.cuda.get_device_name(),cases={},
    route_generator_sha256=hashlib.sha256((root.parent/"muse_moe_w4a16/reference.py").read_bytes()).hexdigest(),
    route_provenance="Kernel-task generate_input(bs,ep8,seed,dist,busiest,cuda); serving expert IDs unobserved",
    scopes="One RTX rank. Other pool contributions precomputed; expert reduction is identity. No router, attention, or distributed communication.",
    complete=False)
def save():
    temp=args.out.with_suffix(".temp");temp.write_text(json.dumps(report,indent=2));temp.replace(args.out)
raw=torch.nn.Module();h=4096;e=32;torch.manual_seed(args.seed)
for stage,n in (("w13",2*h),("w2",h)):
    for name,v in (("weight",torch.randint(0,256,(e,n,h//2),device="cuda",dtype=torch.uint8)),
        ("weight_scale",torch.randint(40,68,(e,n,h//16),device="cuda",dtype=torch.uint8).view(torch.float8_e4m3fn)),
        ("weight_scale_2",torch.linspace(.125,.5,e,device="cuda",dtype=torch.bfloat16))):
        setattr(raw,stage+"_"+name,torch.nn.Parameter(v,requires_grad=False))
with cached_threshold_for_preparation(here):layer=ns["_prepare_humming_experts"](raw,torch)
del raw,v
norm=torch.linspace(.7,1.2,h,device="cuda",dtype=torch.bfloat16)

class Markers:
    def __init__(self):
        self.pool=[torch.cuda.Event(enable_timing=True,external=True) for _ in range(96)]
        for ev in self.pool:ev.record()
        torch.cuda.synchronize()
        self.reset()
    def reset(self):
        self.ranges=[];self.open=set();self.i=0
    def wrap(self,name,fn):
        def timed(*a,**kw):
            label=kw["sublayer_name"] if name=="gemm" else name
            if label in self.open:return fn(*a,**kw)
            start,end=self.pool[self.i:self.i+2];self.i+=2
            self.open.add(label);start.record()
            try:return fn(*a,**kw)
            finally:
                end.record();self.open.remove(label);self.ranges.append((label,start,end))
        return timed
    def components(self):
        summed=collections.Counter()
        for key,a,b in self.ranges:summed[key]+=a.elapsed_time(b)*1000
        # The RMS callable is nested solely in DenseEnvelope.before. All other ranges are exclusive.
        summed["pool_project_cast_assemble"]-=summed["pre_expert_rms"]
        return dict(summed)
    def patches(self,envelope):
        stack=contextlib.ExitStack()
        for obj,name,label in ((HummingMethod,"forward_layer","gemm"),(scale,"scale_active_rows","w13_scaling"),
                (activation,"silu_and_mul","silu"),(align,"moe_align_block_size","alignment"),
                (split,"partition_routes","partition"),(split,"partition_routes_scalable","partition"),
                (torch.nn.functional,"rms_norm","pre_expert_rms"),(envelope,"before","pool_project_cast_assemble"),
                (envelope,"after","unpool_project_cast")):
            f=self.wrap(label,getattr(obj,name))
            stack.enter_context(patch.object(obj,name,staticmethod(f) if obj is HummingMethod else f))
        return stack

def case(m,dist):
    env=DenseEnvelope(m,args.seed);mode="graph" if m<16384 else "eager_serial"
    count=8 if mode=="graph" else 6
    fixture_seed=args.seed+13*m+(0 if dist=="normal" else 7)
    data=ref.generate_input(m,"ep8",fixture_seed,dist=dist,rank="busiest",device="cuda")
    global_ids=data["topk_ids"];start=int(data["config"]["local_start"]);idx=global_ids-start
    ids=torch.where((idx>=0)&(idx<32),idx,-1).to(torch.int32).contiguous()
    scores=data["topk_weights"].contiguous()
    counts=torch.bincount(ids[ids>=0].long(),minlength=32).cpu().tolist()
    del data,idx,global_ids
    torch.manual_seed(fixture_seed);x=torch.empty((m,8192),device="cuda",dtype=torch.bfloat16).normal_(std=.023)
    for offset in range(0,m,1024):
        env.assembled[offset:offset+1024].copy_(torch.mm(x[offset:offset+1024],env.pool.t(),out_dtype=torch.float32))
    torch.cuda.empty_cache()
    rec=report["cases"][f"M{m}_{dist}"]=dict(rows=m,dist=dist,fixture_seed=fixture_seed,mode=mode,
            rank_histogram=counts,local_pairs=sum(counts),max_count=max(counts),active_experts=sum(c>0 for c in counts),
            projection=env.kind,free_bytes_at_start=int(torch.cuda.mem_get_info()[0]),samples=[],complete=False)
    traces=Markers()
    def run(marked):
        inner=env.before(x)
        y,loc=ns["_humming_expert_outputs"](layer,inner,ids,torch,global_experts=256,defer_downscale=True)
        finalize=traces.wrap("post_expert_norm_scale_and_sum",finish) if marked else finish
        result=finalize(y,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=loc,expert_output_scale=layer.w2_post_scale)
        return env.after(result)
    plain=run(False);torch.cuda.synchronize()
    event1,event2=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    events=[];graphs={};outs={}
    if mode=="graph":
        for marked in (False,True):
            traces.reset()
            with traces.patches(env) if marked else contextlib.nullcontext():
                with torch.cuda.graph(g:=torch.cuda.CUDAGraph()):outs[marked]=run(marked)
                graphs[marked]=g
            graphs[marked].replay();torch.cuda.synchronize()
            assert torch.equal(plain,outs[marked])
    else:
        with traces.patches(env):probe=run(True)
        torch.cuda.synchronize();assert torch.equal(plain,probe);del probe
    for i in range(count):
        sample={}
        for marked in ((False,True) if i%2==0 else (True,False)):
            ctx=traces.patches(env) if marked and mode!="graph" else contextlib.nullcontext()
            with ctx:
                if marked and mode!="graph":traces.reset()
                torch.cuda.synchronize()
                wall=time.perf_counter_ns();event1.record()
                if mode=="graph":graphs[marked].replay()
                else:outs[marked]=run(marked)
                event2.record();event2.synchronize()
                wall=(time.perf_counter_ns()-wall)/1000
            label="marked" if marked else "plain"
            sample[label+"_stream_us"]=event1.elapsed_time(event2)*1000
            sample[label+"_wall_us"]=wall
            if marked:sample["components_us"]=traces.components()
        sample["parity"]=torch.equal(outs[True],outs[False]) and bool(torch.isfinite(outs[True]).all())
        assert sample["parity"]
        sample["component_sum_us"]=sum(sample["components_us"].values())
        sample["unattributed_us"]=sample["marked_stream_us"]-sample["component_sum_us"]
        rec["samples"].append(sample);save()
    rec["median_us"]={k:med(r[k] for r in rec["samples"]) for k in ("marked_stream_us","plain_stream_us","plain_wall_us","marked_wall_us","component_sum_us","unattributed_us")}
    rec["median_components_us"]={k:med(r["components_us"].get(k,0) for r in rec["samples"]) for k in rec["samples"][0]["components_us"]}
    rec["complete"]=True;save()
    print(m,dist,sum(counts),rec["median_us"],rec["median_components_us"],flush=True)
save()
for m in args.rows:
    for dist in ("normal","skewed"):
        try:
            case(m,dist)
        except torch.OutOfMemoryError as error:
            if m<16384:raise
            report["cases"].setdefault(f"M{m}_{dist}",{}).update(complete=False,oom=str(error))
            save()
        torch.cuda.empty_cache()
report["complete"]=True;save()


