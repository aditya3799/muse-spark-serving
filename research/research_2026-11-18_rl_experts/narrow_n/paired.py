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
parser.add_argument("--rows",type=int,nargs="+",default=[128,768])
parser.add_argument("--variants",nargs="+",choices=["w13","w2","both"],default=["w13","w2","both"])
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

from policy import narrow
from humming.kernel.humming import HummingKernel
from collections import Counter
variants=args.variants
assert set(variants)<=set(["w13","w2","both"]) and variants
sm=10*torch.cuda.get_device_capability()[0]+torch.cuda.get_device_capability()[1]
original=HummingMethod.forward_layer
report.update(candidate="Full-K CTA-N 128",variants=variants,stage_parity=[],kernels=[],benchmarks={})
seen={};captured=[];role="parent";capture=False
def wrap(layer,**kw):
    stage=kw["sublayer_name"];old=kw["tuning_config"]
    conf=narrow(old,sm,kw["valid_shape_m"],stage,role)
    if capture:captured.append(dict(kw))
    if role!="parent" and role in (stage,"both") and conf is old:
        raise ValueError(f"unexpected inactive ordered variant {role} {stage} {old}")
    kw["tuning_config"]=conf
    meta=layer.humming_metas[stage]
    handles=HummingKernel.prepare_kernels(meta.to_str(),kw["compute_config"],conf).tolist()
    kid=handles[2];kernel=HummingKernel._id2kernel[kid]
    if (role,stage,kid) not in seen:
        seen[role,stage,kid]=dict(role=role,stage=stage,id=kid,block=kernel.block_shape,warp=kernel.warp_shape,
                sha256=hashlib.sha256(Path(kernel.kernel_filename).read_bytes()).hexdigest())
        report["kernels"]=list(seen.values())
    return original(layer,**kw)
HummingMethod.forward_layer=staticmethod(wrap)
norm=torch.linspace(.7,1.2,h,device="cuda",dtype=torch.bfloat16)
def case(m):
    global role,capture
    env=DenseEnvelope(m,args.seed)
    x=torch.empty((m,8192),device="cuda",dtype=torch.bfloat16);ids=torch.empty((m,8),device="cuda",dtype=torch.int32);scores=torch.empty((m,8),device="cuda")
    def update(seed,dist):
        data=ref.generate_input(m,"ep8",seed,dist=dist,rank="busiest",device="cuda")
        raw=data["topk_ids"]-data["config"]["local_start"];ids.copy_(torch.where((raw>=0)&(raw<32),raw,-1));scores.copy_(data["topk_weights"])
        torch.manual_seed(seed);x.normal_(std=.023);env.stage_peers(env.peer_fixture(x))
        return torch.bincount(ids[ids>=0].long(),minlength=32).cpu().tolist()
    def run():
        inner=env.before(x)
        y,loc=ns["_humming_expert_outputs"](layer,inner,ids,torch,global_experts=256,defer_downscale=True)
        return env.after(finish(y,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=loc,expert_output_scale=layer.w2_post_scale))
    update(args.seed,"normal")
    # Independently execute each stage on identical stored route metadata, including all inactive outputs.
    role="parent";capture=True;capture_out=run();capture=False;del capture_out
    for kw in captured:
        stage=kw["sublayer_name"]; valid=int(kw["num_tokens_padded"][0]);logs=[]
        if valid==0:continue
        local=dict(kw);local["inputs"]=torch.empty_like(kw["inputs"])
        pairs=[torch.empty_like(kw["outputs"]) for _ in range(2)]
        for pattern in ("random","mixed_exponent"):
            xx=local["inputs"];torch.manual_seed(args.seed+199);xx.normal_(std=.35)
            if pattern=="mixed_exponent":
                xx[:,::16]*=2.**16;xx[:,1::16]*=2.**-15;xx[:,32::64]*=-1
            for name,buf in zip(("parent",stage),pairs):
                role=name;buf.fill_(123);local["outputs"]=buf;wrap(layer,**local)
            equal=torch.equal(pairs[0].view(torch.int16),pairs[1].view(torch.int16))
            logs.append(dict(pattern=pattern,exact=equal));assert equal,(m,stage,pattern)
        report["stage_parity"].append(dict(rows=m,stage=stage,padded=valid,patterns=logs))
        del pairs,local,xx,buf
    captured.clear()
    graphs={};outs={}
    for name in ["parent"]+variants:
        role=name;run();torch.cuda.synchronize()
        with torch.cuda.graph(g:=torch.cuda.CUDAGraph()):outs[name]=run()
        graphs[name]=g;g.replay()
    for dist in ("normal","skewed"):
        name=f"M{m}_{dist}"
        rec=report["benchmarks"][name]=dict(times={k:[] for k in ["parent","control"]+variants},parity=[],order=[],counts=[])
        for i in range(8):
            rec["counts"].append(update(args.seed+251*m+37*i+(1009 if dist=="skewed" else 0),dist))
            labels=["parent"]+variants[i%len(variants):]+variants[:i%len(variants)]+["control"]
            if i%2:labels.reverse()
            rec["order"].append(labels)
            for name in labels:
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();graphs["parent" if name=="control" else name].replay();b.record();b.synchronize();rec["times"][name].append(a.elapsed_time(b)*1000)
            ok=all(torch.equal(outs[v].view(torch.int16),outs["parent"].view(torch.int16)) for v in variants)
            rec["parity"].append(ok);save();assert ok,(m,dist)
        print(m,dist,{key:round(med(rec["times"]["parent"])/med(values),4) for key,values in rec["times"].items()},flush=True)
save()
for m in args.rows:
    case(m);torch.cuda.empty_cache()
report["complete"]=True;save()

