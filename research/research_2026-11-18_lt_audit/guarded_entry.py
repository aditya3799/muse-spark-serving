"""Replay exact patched startup + project_target_hidden methods on the public draft checkpoint.

Unrelated worker initialization hooks and target transformer layers are supplied as fixtures.
"""
import argparse
import ast
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import torch
from torch.nn import functional as F
import coupled_full
from real_coupled import checkpoint,configured_rope
from audit_match import HERE,N,K,BASELINE,mismatches

ROOT=HERE.parent
PKG=HERE/"integrated/python/sglang/srt/speculative/full_projector"
PATCH=HERE/"integrated_candidate.diff"
PATCH_SHA="84eff9bc0fd443eba15e3b2f23ca8985e7776c562f6d5746255613df5924ce18"
PIN="297e54fd105e974cdf401e5b4d20253a22c16f90"
SEEDS={"pilot":890761,"holdout":1176971}
MODEL="python/sglang/srt/models/dflash.py"
WORKER="python/sglang/srt/speculative/dflash_worker_v2.py"
S=types.SimpleNamespace


def import_package():
    for name in ("sglang","sglang.srt","sglang.srt.speculative"):
        if name not in sys.modules:
            parent=types.ModuleType(name)
            parent.__path__=[]
            sys.modules[name]=parent
    name="sglang.srt.speculative.full_projector"
    spec=importlib.util.spec_from_file_location(name,PKG/"__init__.py",submodule_search_locations=[str(PKG)])
    m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m
    spec.loader.exec_module(m)


def method(path,cls,fn,scope,patched):
    txt=((HERE/"integrated"/path).read_text(encoding="utf-8") if patched else
         subprocess.check_output(["git","-C",str(ROOT/"sglang"),"show",PIN+":"+path]).decode())
    tree=ast.parse(txt)
    owner=next(n for n in tree.body if getattr(n,"name",None)==cls)
    node=next(n for n in owner.body if getattr(n,"name",None)==fn)
    exec(compile(ast.Module(body=[node],type_ignores=[]),path,"exec"),scope)
    return scope[fn]


def model_method(patched):
    return method(MODEL,"DFlashDraftModel","project_target_hidden",{"torch":torch},patched)


def startup(model,enabled,batches):
    group=S(world_size=1)
    workspace=S(draft_model=model,draft_parallel_layout=None,block_size=8,
                model_runner=S(prewarm_sampling=lambda:"existing sampling warmup"))
    config=S(deterministic=S(enable_deterministic_inference=False),
             graph=S(cuda_graph_config=S(decode=S(bs=batches))))
    fn=method(WORKER,"DFlashWorkerV2","prewarm_sampling",dict(
        SamplingPrewarmResult=object,is_cuda=lambda:enabled,get_exec=lambda:config,
        is_dflash_sampling_verify_available=lambda:False,draft_tp_context=lambda _:nullcontext(),
        get_parallel=lambda:S(tp_group=group),torch=torch),True)
    assert fn(workspace)=="existing sampling warmup"


class NativeNorm:
    def __init__(self,weight,kernel):
        self.weight,self.kernel=weight,kernel
        self.out,self.last_fc=None,None
    def __call__(self,projection):
        self.last_fc=projection
        out=self.out if self.out is not None else torch.empty_like(projection)
        self.kernel(projection,self.weight,out)
        return out


class ProductionModel:
    def __init__(self,fc,norm,method):
        self.fc,self.hidden_norm=fc,norm
        self.quantizable=False
        self.project_target_hidden=types.MethodType(method,self)


class Admission:
    def __init__(self,model,rows):
        self.model,self.rows=model,rows
        self.reason="native passthrough"
        self.full_bytes=None
        self.initial_bad=None
        self.index=None
    def check(self,x):
        if self.rows<128: return
        runner=getattr(self.model,"_context_projector",None)
        if runner is None: return
        decision=runner.entries.get(runner.key(x))
        if decision is None: return
        entry,_=decision
        self.full_bytes=entry[2].numel()
        self.reason="runtime plan applied"
        self.index=entry[1]
        self.initial_bad=mismatches(runner(x),self.model.fc(x))


def cpu():
    import_package()
    original,patched=model_method(False),model_method(True)
    fc=torch.nn.Linear(32,16,bias=False,dtype=torch.bfloat16).eval()
    class Guard(ProductionModel):
        def __getattribute__(self,n):
            if n=="_context_projector" and object.__getattribute__(self,"ban"):
                raise AssertionError("small input performed a projector lookup")
            return super().__getattribute__(n)
    m=Guard(fc,lambda v:v,patched)
    m.num_context_features=2
    m.config=S(hidden_size=16)
    m.ban=True
    for rows in (7,127):
        x=torch.randn(rows,32,dtype=torch.bfloat16)
        assert mismatches(m.project_target_hidden(x),original(m,x))==0
    m.ban=False
    assert torch.equal(m.project_target_hidden(torch.ones(128,32,dtype=torch.bfloat16)),fc(torch.ones(128,32,dtype=torch.bfloat16)))
    try: m.project_target_hidden(torch.zeros(2,31))
    except ValueError: pass
    else: raise AssertionError("invalid geometry unexpectedly accepted")
    startup(m,False,[2,33])
    assert not torch.cuda.is_initialized()
    print("actual patched cold/small entry, large native fallback, original validation and startup CPU fallback passed")


def run(args):
    assert hashlib.sha256(PATCH.read_bytes()).hexdigest()==PATCH_SHA
    if args.stage=="cpu": return cpu()
    import_package()
    (fc,kv,norm_weight,k_norm),weight_report=checkpoint("cuda")
    rope=configured_rope("cuda",weight_report["rope_theta"])
    rms=coupled_full.rms_linux()
    fc_layer=torch.nn.Linear(K,N,bias=False,device="meta",dtype=torch.bfloat16)
    fc_layer.weight=torch.nn.Parameter(fc,requires_grad=False)
    fc_layer.eval()
    original,patched=model_method(False),model_method(True)
    participants={arm:ProductionModel(fc_layer,NativeNorm(norm_weight,rms),patched if arm=="candidate" else original)
                  for arm in coupled_full.ARMS}
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        start=time.perf_counter()
        startup(participants["candidate"],True,[1,4,16,32])
        torch.cuda.synchronize()
        cold_s=time.perf_counter()-start
        runner=getattr(participants["candidate"],"_context_projector",None)
        admitted=sorted(key[1] for key in runner.entries) if runner is not None else []
        setup={"startup_s":cold_s,"admitted_rows":admitted,"native_below_rows":128,"patch_sha256":PATCH_SHA,
               "single_physical_rank":True,"scheduler_forward_stream_is_nondefault":stream.cuda_stream!=torch.cuda.default_stream().cuda_stream}
        case=coupled_full.patched(checkpoint_k_norm=k_norm,track_changes=True,entry_models=participants)
        for stage in (list(SEEDS) if args.stage=="both" else [args.stage]):
            doc={"stage":stage,"seed":SEEDS[stage],"details":{},"startup":setup,"weights":weight_report,
                 "actual_worker_method":"DFlashWorkerV2.prewarm_sampling","actual_projection_method":"DFlashDraftModel.project_target_hidden",
                 "synthetic_target_taps":True}
            canonical=dict(agent="speculation",candidate="guarded_production_real_projection",scope="scored-parent context subset on one SM86; actual staged startup and project_target_hidden methods; public DFlash2 FC, RMS, TP0 K/V and RoPE",
                baseline_sha256=BASELINE,candidate_sha256=PATCH_SHA,correct=True,active=bool(admitted),
                stage=stage,seed=SEEDS[stage],bench={"instances":1,"shapes":{}})
            for b in (1,4,16,32):
                for mode in ("partial","full"):
                    name=f"B{b}_{mode}"
                    test=Admission(participants["candidate"],b*8)
                    row=case(b,SEEDS[stage],fc,kv,norm_weight,rope,rms,args.rounds,test,mode)
                    row["initial_selected_plan_bad"]=test.initial_bad
                    row["activated"]=test.index is not None
                    row["all_changes_positive"]=min(row["changed_projection_values"])>0
                    doc["details"][name]=row
                    canonical["bench"]["shapes"][name]={"weight":1,"times":row["wall_us"]}
                    canonical["correct"]&=(test.initial_bad in (None,0) and row["all_changes_positive"]
                        and all(v==0 for v in row["wrong"].values())
                        and all(v==0 for stages in row["wrong_stages"].values() for v in stages.values()))
                    print(stage,name,row["disposition"],row["all_changes_positive"],canonical["correct"],flush=True)
                    (HERE/f"guarded_{stage}_detail.json").write_text(json.dumps(doc,indent=2)+"\n")
                    (HERE/f"guarded_{stage}.json").write_text(json.dumps(canonical,indent=2)+"\n")
                    torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if runner is not None: runner.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("cpu","pilot","holdout","both"))
    p.add_argument("--rounds",type=int,default=12)
    with torch.inference_mode(): run(p.parse_args())

