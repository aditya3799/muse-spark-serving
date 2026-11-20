"""Pair the actual production owner and general composite, including rank remapping."""
import argparse
import ast
import hashlib
import json
import statistics
import sys
import types
from pathlib import Path
from typing import Optional
here=Path(__file__).resolve().parent
ref_src=(here/"paired_fused.py").read_text()
setup=ref_src[:ref_src.index("import grouped_fused as grouped")]
setup=setup.replace('args=parser.parse_args()','parser.add_argument("--serial",action="store_true");args=parser.parse_args()')
exec(compile(setup,str(here/"general_paired.py"),"exec"),globals())
from types import SimpleNamespace
from sglang.kernels.ops.moe import muse_spark_humming_split as frozen_split
parent_helpers=ns
qn="sglang.kernels.ops.moe."
hn="sglang.srt.layers.muse_spark.muse_spark_nvfp4_moe"
on="srt/layers/muse_spark/muse_spark_fp4_moe.py"
spec=fixture.module
regime=spec(qn+"muse_spark_route_policy",here/"general_muse_spark_route_policy.py")
group=spec(qn+"muse_spark_grouped_routes",here/"general_muse_spark_grouped_routes.py")
act=spec(qn+"route_activation",here/"general_route_activation.py")
candidate_split=spec(qn+"muse_spark_humming_split",here/"general_muse_spark_humming_split.py")
owner_path=fixture.SOURCE/on
def load_owner(src):
    names={"_marlin_expert_outputs","_forward_marlin"}
    fns=[n for o in ast.parse(src).body if isinstance(o,ast.ClassDef) for n in o.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(fns)==2
    for f in fns:f.decorator_list=[]
    scope=dict(torch=torch,nn=torch.nn,Optional=Optional,fused_post_expert_norm_weighted_sum=finish)
    exec(compile(ast.Module(body=fns,type_ignores=[]),"owner","exec"),scope)
    return scope
owners=[load_owner(owner_path.read_text()),load_owner((here/"general_muse_spark_fp4_moe.py").read_text())]
helper_file=here/"general_muse_spark_nvfp4_moe.py"
selected=[n for n in ast.parse(helper_file.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in
          ("_humming_expert_outputs","_humming_expert_outputs_default")]
cs=dict(torch=torch);exec(compile(ast.Module(body=selected,type_ignores=[]),str(helper_file),"exec"),cs)
helpers=[SimpleNamespace(_humming_expert_outputs=parent_helpers["_humming_expert_outputs"]),SimpleNamespace(_humming_expert_outputs=cs["_humming_expert_outputs"])]
pkg_name=qn+"muse_spark_humming_split"
import sglang.kernels.ops.moe as moe_package
stats=dict(grouped=0,group8=0,ordinary=0)
orig_group=group.compute
def counted_group(*a,**kw):
    stats["grouped"]+=1
    return orig_group(*a,**kw)
group.compute=counted_group
orig_scaled=act.scaled_prefill_activation
def reached(*a,**kw):
    dense=kw.get("dense_routes",False)
    if not dense:stats["ordinary"]+=1
    return orig_scaled(*a,**kw)
act.scaled_prefill_activation=reached
orig_load=act.load_jit
def relocate(*a,**kw):
    assert kw["cuda_files"]==["elementwise/route_activation.cuh"]
    kw["cuda_files"]=[str(root/"research_2026-11-18_rl_prefill/route_activation.cuh")]
    return orig_load(*a,**kw)
act.load_jit=relocate
get_native=act.route_activation_module
class CountNative:
    def run(self,*a):
        stats["group8"]+=1
        get_native().run(*a)
act.route_activation_module=lambda:CountNative()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rec=dict(agent="experts",candidate="tile_occupancy_grouped_or_group8",scope="Actual frozen and proposed production Humming FFN owner including rank mapping, post norm and local projections; GPU RTX3050, synthetic busiest EP8 official kernel-task routes; no collectives/attention.",
    baseline_sha256=sha(fixture.RELEASE/"candidate.diff"),candidate_sha256=sha(here/"general_joint.diff"),seed=args.seed,
    correct=True,active=False,complete=False,bench=dict(instances=1,shapes={}),dispatch={},smokes=[],serial=args.serial)
def save():
    tmp=args.out.with_suffix(".temp");tmp.write_text(json.dumps(rec,indent=2));tmp.replace(args.out)
def arm(which):
    split_mod=[frozen_split,candidate_split][which]
    sys.modules[pkg_name]=split_mod
    setattr(moe_package,"muse_spark_humming_split",split_mod)
    sys.modules[hn]=helpers[which]
def case(m):
    print("case",m,flush=True)
    env=DenseEnvelope(m,args.seed)
    x=torch.empty((m,8192),device="cuda",dtype=torch.bfloat16)
    gids=torch.empty((m,8),device="cuda",dtype=torch.int32);scores=torch.empty((m,8),device="cuda")
    lookup=torch.full((257,),-1,device="cuda",dtype=torch.int32);lookup[33:65]=torch.arange(32,device="cuda",dtype=torch.int32)
    normobj=object()
    owner=SimpleNamespace(use_humming=True,global_num_experts=256,_post_expert_norm=normobj,
        _post_expert_norm_weight_first=True,_post_expert_norm_eps=1e-8,_post_expert_norm_weight=norm,
        _expert_map=lookup[1:],_expert_lookup=lookup,_humming_layer=layer)
    def update(seed,dist):
        d=ref.generate_input(m,"ep8",seed,dist="normal" if dist=="all_remote" else dist,rank="busiest",device="cuda")
        # Rotate the actual busiest shard into global rank 1.
        gids.copy_((d["topk_ids"]-d["config"]["local_start"]+32)%256)
        if dist=="all_remote":gids.fill_(188)
        scores.copy_(d["topk_weights"]);torch.manual_seed(seed);x.normal_(std=.023)
        env.stage_peers(env.peer_fixture(x))
    def run(i):
        arm(i)
        owner._marlin_expert_outputs=types.MethodType(owners[i]["_marlin_expert_outputs"],owner)
        result=owners[i]["_forward_marlin"](owner,env.before(x),scores,gids,normobj)
        return env.after(result)
    update(args.seed,"normal")
    graphs=[];outputs=[None,None]
    for i in range(2):
        before=dict(stats);outputs[i]=run(i);torch.cuda.synchronize()
        if not args.serial:
            with torch.cuda.graph(g:=torch.cuda.CUDAGraph()):outputs[i]=run(i)
            graphs.append(g);g.replay()
        if i:
            calls={k:stats[k]-before[k] for k in stats}
            rec["dispatch"][str(m)]=calls;print("dispatch",m,calls,flush=True)
            if calls["grouped"] or calls["group8"]:rec["active"]=True
    def execute(i):
        if args.serial:outputs[i]=run(i)
        else:graphs[i].replay()
    def exact():
        ok=torch.equal(outputs[0].view(torch.int16),outputs[1].view(torch.int16))
        if not ok:rec.update(correct=False,complete=True);save();raise AssertionError("full FFN bitwise mismatch")
        return ok
    for dist in ("all_remote","normal","skewed"):
        update(args.seed+83*m,dist);execute(0);execute(1)
        rec["smokes"].append(dict(m=m,dist=dist,exact=exact()));save()
    for dist in ("normal","skewed"):
        row=rec["bench"]["shapes"][f"M{m}_{dist}"]=dict(weight=1,times={k:[] for k in ("baseline","candidate","control")},parity=[],orders=[])
        for j in range(args.rounds):
            update(args.seed+199*m+73*j+(2503 if dist=="skewed" else 0),dist)
            labels=("baseline","candidate","control") if j%2==0 else ("control","candidate","baseline")
            row["orders"].append(labels)
            for label in labels:
                i=int(label=="candidate");arm(i)
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();execute(i);b.record();b.synchronize();row["times"][label].append(a.elapsed_time(b)*1000)
            row["parity"].append(exact());save()
        print(m,dist,round(statistics.median(row["times"]["baseline"])/statistics.median(row["times"]["candidate"]),4),flush=True)
for m in args.rows:
    case(m);torch.cuda.empty_cache()
rec["complete"]=True;save()
