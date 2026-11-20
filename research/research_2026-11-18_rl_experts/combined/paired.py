"""Three-arm graph replay on exact independently materialized candidate sources."""
import argparse
import ast
import collections
import hashlib
import json
import statistics
import sys
from pathlib import Path

here=Path(__file__).resolve().parent
root=here.parents[1]
parser=argparse.ArgumentParser()
parser.add_argument("--seed",type=int,required=True)
parser.add_argument("--out",type=Path,required=True)
parser.add_argument("--rows",type=int,nargs="+",default=[8,32,128,256,768,3072])
parser.add_argument("--pairs",type=int,default=8)
args=parser.parse_args()
assert args.seed>0 and args.pairs>=8 and args.pairs%2==0
sys.path[:0]=[str(root/"research_2026-11-18_target_moe"),str(here.parent)]
import bench_experts as fixture
for path in (fixture.OLD/"humming_vendor",fixture.OLD/"phase2",root/"optimization_marlin_tiles_2026-11-15"):
    sys.path.insert(0,str(path))
from kernel_runtime import bootstrap
bootstrap(root/"sglang")
import torch
from dense_envelope import DenseEnvelope
from fixture_tuning_cache import cached_threshold_for_preparation
key="sglang.kernels.ops.moe.muse_spark_humming_split"
mod="sglang.kernels.ops.moe."
moe=fixture.SOURCE/"kernels/ops/moe"
fixture.module(mod+"muse_spark_humming_scale",moe/"muse_spark_humming_scale.py")
finish=fixture.module("combined_final",moe/"muse_spark_post_expert_finalize_triton.py").fused_post_expert_norm_weighted_sum
order=fixture.module(mod+"scale_order",here/"combined/scale_order.py")
activation=fixture.module(mod+"route_activation",here/"combined/route_activation.py")
load=activation.load_jit
def relocated(*args,**kwargs):
    assert kwargs["cuda_files"]==["elementwise/route_activation.cuh"]
    kwargs["cuda_files"]=[str(here/"combined/route_activation.cuh")]
    return load(*args,**kwargs)
activation.load_jit=relocated
binary=activation.route_activation_module()
calls=collections.Counter()
role=None
class Traced:
    def run(self,*args):
        calls[role,"ballot_native"]+=1
        return binary.run(*args)
activation.route_activation_module=lambda:Traced()
original_scale=activation.scaled_prefill_activation
def trace_scale(*args):
    calls[role,"activation_wrapper"]+=1
    return original_scale(*args)
activation.scaled_prefill_activation=trace_scale
sys.path.insert(0,str(here.parent/"upstream_scale_order"))
from scale_evidence import observe
manifest=json.loads((here/"composition.json").read_text())
report=dict(agent="experts",candidate="scale_register_order_and_group8_activation",
    baseline_sha256=manifest["inputs"]["candidate.diff:release_2026-11-18"],candidate_sha256=manifest["candidate_sha256"],
    seed=args.seed,unit="us",gpu=torch.cuda.get_device_name(),correct=True,active=False,complete=False,
    scope="Whole rank-local routed MoE and TP8-shaped FFN graphs; missing peer reductions staged from synthetic fixtures. Cases have equal test weight.",
    bench={"instances":1,"shapes":{}},incremental={},dispatch={})
observe(order,report)
native_forward=order.scale_order_forward
def trace_forward(layer,**kw):
    before=order.kernel_handle.cache_info()
    out=native_forward(layer,**kw)
    after=order.kernel_handle.cache_info()
    if after.hits+after.misses>before.hits+before.misses:
        calls[role,kw["sublayer_name"]]+=1
    return out
order.scale_order_forward=trace_forward
def save():
    temp=args.out.with_suffix(".temp");temp.write_text(json.dumps(report,indent=2));temp.replace(args.out)
def extract(path):
    names={"_prepare_humming_experts","_humming_expert_outputs_default","_humming_expert_outputs"}
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns={"torch":torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),ns);return ns
arms={}
for name in ("parent","ballot","combined"):
    path=here/name
    arms[name]=(fixture.module(name+"_combined_split",path/"muse_spark_humming_split.py"),extract(path/"muse_spark_nvfp4_moe.py"))
sys.modules[key]=arms["parent"][0]
raw=torch.nn.Module();h=4096;e=32
torch.manual_seed(args.seed)
finite=torch.tensor([v for v in range(256) if v not in (127,255)],device="cuda",dtype=torch.uint8)
for stage,n in (("w13",2*h),("w2",h)):
    gs=torch.randint(40,68,(e,n,h//16),device="cuda",dtype=torch.uint8);gs[:,:254,0].copy_(finite)
    for name,v in (("weight",torch.randint(0,256,(e,n,h//2),device="cuda",dtype=torch.uint8)),
            ("weight_scale",gs.view(torch.float8_e4m3fn)),("weight_scale_2",torch.linspace(.125,.5,e,device="cuda",dtype=torch.bfloat16))):
        setattr(raw,stage+"_"+name,torch.nn.Parameter(v,requires_grad=False))
with cached_threshold_for_preparation(here): layer=arms["parent"][1]["_prepare_humming_experts"](raw,torch)
del raw,v,gs
norm=torch.linspace(.7,1.2,h,device="cuda",dtype=torch.bfloat16)
def run_case(m,scope):
    global role
    env=DenseEnvelope(m,args.seed) if scope=="ffn" else None
    x=torch.empty((m,8192 if env else h),dtype=torch.bfloat16,device="cuda")
    ids=torch.empty((m,8),dtype=torch.int32,device="cuda");scores=torch.empty((m,8),device="cuda")
    def update(seed,cluster):
        gen=torch.Generator().manual_seed(seed);w=torch.rand((m,256),generator=gen)
        if cluster: w[::2,:8]+=2
        idx=w.topk(8).indices;ids.copy_(torch.where(idx<32,idx,-1))
        torch.manual_seed(seed);x.normal_(std=.023);scores.uniform_();scores.div_(scores.sum(1,keepdim=True))
        if env:env.stage_peers(env.peer_fixture(x))
    def forward(name):
        global role
        role=name;module,ns=arms[name];sys.modules[key]=module
        inner=env.before(x) if env else x
        y,loc=ns["_humming_expert_outputs"](layer,inner,ids,torch,global_experts=256,defer_downscale=True)
        result=finish(y,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=loc,expert_output_scale=layer.w2_post_scale)
        return env.after(result) if env else result
    update(args.seed+1,False)
    graphs={};outs={};calls.clear()
    for name in arms:
        forward(name);torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): outs[name]=forward(name)
        graphs[name]=g
    assert calls["combined","w13"] and calls["combined","w2"]
    for name in ("ballot","combined"):
        assert calls[name,"activation_wrapper"] and bool(calls[name,"ballot_native"])==(m>=512)
    report["dispatch"][f"{scope}_M{m}"]={f"{a}_{b}":v for (a,b),v in calls.items()}
    for cluster in (False,True):
        case=f"{scope}_M{m}_" + ("cluster" if cluster else "uniform")
        tr=report["bench"]["shapes"][case]=dict(weight=1,times={k:[] for k in ("baseline","candidate","control")},parity=[],order=[])
        inc=report["incremental"][case]=dict(weight=1,times={k:[] for k in ("baseline","candidate","control")},parity=[],order=[])
        for i in range(args.pairs):
            update(args.seed+37*i+113,cluster)
            labels=["parent","combined","ballot","ballot_control","parent_control"]
            if i%2:labels.reverse()
            tmp={}
            for label in labels:
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                arm=label.replace("_control","")
                a.record();graphs[arm].replay();b.record();b.synchronize();tmp[label]=a.elapsed_time(b)*1000
            ok=all(torch.equal(outs["parent"],outs[n]) for n in ("combined","ballot")) and bool(torch.isfinite(outs["parent"]).all())
            for record,reference in ((tr,"parent"),(inc,"ballot")):
                for outkey,label in (("baseline",reference),("candidate","combined"),("control",reference+"_control")):
                    record["times"][outkey].append(tmp[label])
                record["order"].append(labels);record["parity"].append(ok)
            report["correct"]&=ok;save();assert ok,case
        print(case,round(statistics.median(tr["times"]["baseline"])/statistics.median(tr["times"]["candidate"]),4),round(statistics.median(inc["times"]["baseline"])/statistics.median(inc["times"]["candidate"]),4),flush=True)
save()
for rows in args.rows:
    for scope in ("moe","ffn"):
        run_case(rows,scope);torch.cuda.empty_cache()
report.update(active=True,complete=True);save()

