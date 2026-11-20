"""Coupled pinned residual taps, exact FlashInfer RMS and native draft KV; replace only F.linear.

Frozen algorithm descriptors are loaded from a distinct discovery run for this evaluation.
Deployment must rediscover on its own host; there is no compiled model-specific config.
"""
import argparse
import ctypes
import inspect
import json
import random
import statistics
import sys
import threading
from pathlib import Path
import torch
from torch.nn import functional as F
from audit_match import N,K,BASELINE,describe,load_lib,mismatches
from runtime_projection import native_workspace
HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/"research_2026-11-18_rl_dflash"
sys.path.insert(0,str(PRIOR))
import joint_bench as pinned
from torch.nn import functional as F

STAGES={"pilot":1315217,"holdout":863423}
ARMS=("baseline","candidate","control")
ENV={**pinned.__dict__,"ARMS":ARMS}
FIELDS="id tile split reduction swizzle custom stages inner cluster".split()


class FrozenFull:
    def __init__(self,weights,m,frozen):
        self.lib=load_lib()
        budget=native_workspace()[1]
        self.cap=max(budget,32*2**20)
        self.q=self.lib.make_sweep(m,N,N,K,N,budget,self.cap,8)
        self.index=None
        self.reason="configuration unavailable"
        self.full_bytes=None
        self.w=weights
        self.m=m
        if not self.q: return
        for i in range(self.lib.sweep_count(self.q)):
            desc=describe(self.lib,self.q,i)
            if all(desc[f]==frozen[f] for f in FIELDS):
                self.index=i
                self.full_bytes=desc["workspace"][1]
                # The actual algorithm's checked minimum; graph-safe per stream.
                self.temp=torch.empty(self.full_bytes,device="cuda",dtype=torch.uint8)
                self.reason="applied"
                break
    def execute(self,x,w):
        if self.index is None: return F.linear(x,w)
        out=torch.empty((self.m,N),device=x.device,dtype=x.dtype)
        rc=self.lib.sweep_call(self.q,self.index,1,w.data_ptr(),x.data_ptr(),out.data_ptr(),
            self.temp.data_ptr() if self.full_bytes else None,self.full_bytes,torch.cuda.current_stream().cuda_stream)
        if rc: raise RuntimeError(str(rc))
        return out
    def check(self,x):
        if self.index is None: return
        if mismatches(self.execute(x,self.w),F.linear(x,self.w)):
            self.index=None
            self.reason="actual first coupled input disagreed"
    def close(self):
        if self.q: self.lib.drop_sweep(self.q)


def patched(checkpoint_k_norm=None,track_changes=False,entry_models=None):
    source=inspect.getsource(pinned.run_case)
    def replace(before,after):
        nonlocal source
        assert source.count(before)==1,before
        source=source.replace(before,after)
    replace("kv_weight, norm_weight, rope, rms, rounds):","kv_weight, norm_weight, rope, rms, rounds, fcplan, risk):")
    replace('producer = {"original": capture(source_a), "packed": capture(source_b)}',
            'producer = {"original": capture(source_a)}\n    fcplan.check(producer["original"][1][1])\n    contexts = {}')
    replace("context = F.linear(outputs[1], fc)",
            'context = fcplan.execute(outputs[1],fc) if name=="candidate" else F.linear(outputs[1],fc)\n        contexts[name]=context')
    replace("if step % 8 == 0:",'if risk=="full":')
    a=source.index("        # Cyclic Latin square")
    z=source.index("        order_log.append(order)",a)
    source=source[:a]+'''        if step%2==0:
            order=list(ARMS)
            random.Random(seed+b*113+step).shuffle(order)
        else:
            order=list(reversed(order_log[-1]))
'''+source[z:]
    a=source.index('        assert torch.equal(producer["packed"]')
    z=source.index("        # Exact BF16 bits",a)
    source=source[:a]+'''        for name in ARMS[1:]:
            for tag,val,ref in (("projected_fc",contexts[name],contexts["baseline"]),
                                ("rms",streams[name][1],streams["baseline"][1]),
                                ("dense_kv",streams[name][2],streams["baseline"][2])):
                wrong_stages[name][tag]+=mismatches(val,ref)
'''+source[z:]
    replace("    for step in range(rounds):",'    wrong_stages = {arm:dict(projected_fc=0,rms=0,dense_kv=0) for arm in ARMS[1:]}\n    for step in range(rounds):')
    replace("        wrong=wrong,","        wrong=wrong, wrong_stages=wrong_stages, disposition=fcplan.reason, workspace=fcplan.full_bytes,")
    if checkpoint_k_norm is not None:
        replace("norm = rand((LAYERS, DIM), torch.bfloat16)","norm = checkpoint_k_norm")
    if track_changes:
        replace("    for step in range(rounds):","    prev_context=None\n    changes=[]\n    for step in range(rounds):")
        replace("        # Exact BF16 bits","        if prev_context is not None:\n            changes.append(mismatches(prev_context,contexts['baseline']))\n        prev_context=contexts['baseline'].clone()\n        # Exact BF16 bits")
        replace("        wall_us=wall,","        changed_projection_values=changes, wall_us=wall,")
    if entry_models is not None:
        begin=source.index("        context = fcplan.execute")
        end=source.index("        consumer(direct, pool)",begin)
        source=source[:begin]+"""        pool, normalized, _, consumer = streams[name]
        participant=entry_models[name]
        participant.hidden_norm.out=normalized
        participant.project_target_hidden(outputs[1])
        contexts[name]=participant.hidden_norm.last_fc
"""+source[end:]
    scope={**ENV,"random":random,"mismatches":mismatches,"checkpoint_k_norm":checkpoint_k_norm,"entry_models":entry_models}
    exec(compile(source,"coupled_projector_case.py","exec"),scope)
    return scope["run_case"]


def rms_linux():
    src=inspect.getsource(pinned.native_rms)
    assert src.count('HERE / "rms_flashinfer_072.dll"')==1
    src=src.replace('HERE / "rms_flashinfer_072.dll"','HERE / "rms_linux.so"')
    ns={**ENV,"HERE":HERE}
    exec(src,ns)
    return ns["native_rms"]()


def main(stage,rounds):
    case=patched()
    rms=rms_linux()
    if stage=="cpu":
        print("Pinned original tap/consumer harness, Linux release RMSNorm, and frozen full-width selector loaded")
        return
    assert rounds>1 and rounds%2==0
    seed=STAGES[stage]
    gen=torch.Generator(device="cuda").manual_seed(seed)
    def rand(*shape,std):
        return torch.empty(*shape,device="cuda",dtype=torch.bfloat16).normal_(std=std,generator=gen)
    fc=rand(N,K,std=K**-.5)
    kv=rand(N,1280,std=N**-.5)
    norm=rand(N,std=1.)
    rope=rand(131072,128,std=1.)
    selections=json.loads((HERE/"full_width_detail.json").read_text())
    detail={"seed":seed,"stage":stage,"arms":ARMS,"details":{}}
    canonical=dict(agent="speculation",candidate="coupled_full_width_vendor_projection",
        scope="one SM86 five pinned residual taps, projection, FlashInfer 0.7.2 RMS, dense BF16 draft KV and cache stores; no target transformer or TP transport",
        baseline_sha256=BASELINE,correct=True,active=True,seed=seed,stage=stage,unit="us synchronized wall",
        bench={"instances":1,"shapes":{}})
    for b in [1,4,16,32]:
        for risk in ["partial","full"]:
            name=f"B{b}_{risk}"
            plan=FrozenFull(fc,b*8,selections[str(b*8)]["chosen"])
            try:
                state=case(b,seed,fc,kv,norm,rope,rms,rounds,plan,risk)
                detail["details"][name]=state
                canonical["bench"]["shapes"][name]={"weight":1,"times":state["wall_us"]}
                canonical["active"]&=state["disposition"]=="applied"
                canonical["correct"]&=all(v==0 for v in state["wrong"].values())
                canonical["correct"]&=all(v==0 for stages in state["wrong_stages"].values() for v in stages.values())
                print(stage,name,state["disposition"],{k:round(statistics.median(v),1) for k,v in state["wall_us"].items()},state["wrong_stages"],flush=True)
                (HERE/f"coupled_{stage}_detail.json").write_text(json.dumps(detail,indent=2)+"\n")
                (HERE/f"coupled_{stage}.json").write_text(json.dumps(canonical,indent=2)+"\n")
            finally: plan.close()
            torch.cuda.empty_cache()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["cpu",*STAGES])
    p.add_argument("--rounds",type=int,default=12)
    a=p.parse_args()
    with torch.inference_mode(): main(a.stage,a.rounds)

