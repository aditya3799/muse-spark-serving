"""Coupled test through frozen full-work plans with real DFlash2 FC, K/V and normalizations."""
import argparse
import ast
import hashlib
import json
import math
import os
import statistics
import subprocess
import types
from pathlib import Path
from typing import Union

import torch
from torch.nn import functional as F
from audit_match import BASELINE,HERE,K,N,mismatches
import coupled_full

SOURCE=HERE.parent/"research_2026-11-18_rl"
PINNED="297e54fd105e974cdf401e5b4d20253a22c16f90"
REV="46c90f412549c472941a68382d91a99e5e6372a0"
SEEDS={"pilot":1571017,"holdout":917091}
TYPES={"BF16":torch.bfloat16,"F32":torch.float32,"F16":torch.float16}


def native_path(name):
    if os.name=="nt": return Path(name)
    converted=Path(subprocess.check_output(["wslpath","-a","-u",name],text=True).strip())
    if converted.is_file(): return converted
    # DrvFs on some WSL installations exposes only long Windows file names.
    if "ADITYA~1" in str(converted):
        candidates=list(Path("/mnt/c/Users").glob("*/AppData/Local/Temp/"+converted.name))
        if len(candidates)==1: return candidates[0]
    raise FileNotFoundError(converted)


def checkpoint(device):
    f=json.loads((SOURCE/"public_fc_receipt.json").read_text())
    a=json.loads((SOURCE/"public_aux_receipt.json").read_text())
    assert f["revision"]==a["revision"]==REV
    assert f["dtype"]=="BF16" and f["shape"]==[N,K]
    assert a["rope_parameters"]=={"rope_theta":500000,"rope_type":"default"}
    assert a["rms_norm_eps"]==1e-5
    fp,ap=native_path(f["temporary_file"]),native_path(a["temporary_file"])
    for path,desc in [(fp,f),(ap,a)]:
        assert path.stat().st_size==desc["bytes"]
        with path.open("rb") as source:
            assert hashlib.file_digest(source,"sha256").hexdigest()==desc["sha256"]
    fc=torch.from_file(str(fp),size=N*K,dtype=TYPES[f["dtype"]]).view(N,K)
    payload=bytearray(ap.read_bytes())
    tensors={}
    for name,desc in a["tensors"].items():
        off,size=desc["offset"],desc["bytes"]
        view=memoryview(payload)[off:off+size]
        assert hashlib.sha256(view).hexdigest()==desc["sha256"]
        tensors[name]=torch.frombuffer(view,dtype=TYPES[desc["dtype"]]).view(desc["shape"])
    layers=[f"layers.{i}.self_attn." for i in range(5)]
    kv=torch.cat([tensors[p+tag+".weight"] for p in layers for tag in ("k_proj","v_proj")]).T.contiguous()
    norm=tensors["hidden_norm.weight"]
    k_norm=torch.stack([tensors[p+"k_norm.weight"] for p in layers])
    assert kv.shape==(8192,1280) and norm.shape==(8192,) and k_norm.shape==(5,128)
    assert fc.dtype==kv.dtype==norm.dtype==k_norm.dtype==torch.bfloat16
    return [t.to(device) for t in (fc,kv,norm,k_norm)],dict(revision=REV,fc_sha256=f["sha256"],
        aux_sha256=a["sha256"],runtime_kv_rank=0,rope_theta=a["rope_theta"],rms_norm_eps=a["rms_norm_eps"],
        normalization_dtypes={k:v["dtype"] for k,v in a["tensors"].items() if "norm" in k})


def configured_rope(device,theta):
    filename="python/sglang/srt/layers/rotary_embedding/base.py"
    source=subprocess.check_output(["git","-C",str(HERE.parent/"sglang"),"show",PINNED+":"+filename]).decode()
    tree=ast.parse(source)
    root=next(n for n in tree.body if getattr(n,"name",None)=="RotaryEmbedding")
    keep={"_compute_inv_freq","_compute_cos_sin_cache"}
    methods=[n for n in root.body if getattr(n,"name",None) in keep]
    assert len(methods)==2
    ns={"torch":torch,"Union":Union}
    exec(compile(ast.Module(body=methods,type_ignores=[]),filename,"exec"),ns)
    instance=types.SimpleNamespace(rotary_dim=128,max_position_embeddings=131072,base=theta,position_start=0,_force_native=False)
    instance._compute_inv_freq=types.MethodType(ns["_compute_inv_freq"],instance)
    with torch.device(device):
        rope=ns["_compute_cos_sin_cache"](instance)
    assert rope.dtype==torch.float32 and rope.shape==(131072,128)
    return rope


class FrozenNoSelection(coupled_full.FrozenFull):
    def check(self,x):
        # Record every failure against the unmodified frozen candidate; do not reselect.
        self.initial_bad=None if self.index is None else mismatches(self.execute(x,self.w),F.linear(x,self.w))


def main(args):
    device="cpu" if args.stage=="cpu" else "cuda"
    (fc,kv,norm,k_norm),weights=checkpoint(device)
    rope=configured_rope(device,weights["rope_theta"])
    case=coupled_full.patched(checkpoint_k_norm=k_norm,track_changes=True)
    rms=coupled_full.rms_linux()
    if args.stage=="cpu":
        assert not torch.cuda.is_initialized()
        print("checkpoint hash, sixteen native slices, pinned FP32 RoPE, source transformation and CPU-only checks passed")
        return
    selections=json.loads((HERE/"full_width_detail.json").read_text())
    assert args.rounds>=4 and args.rounds%2==0
    for stage in ([args.stage] if args.stage!="both" else list(SEEDS)):
        seed=SEEDS[stage]
        doc={"seed":seed,"stage":stage,"weights":weights,"selection_source_sha256":hashlib.sha256((HERE/"full_width_detail.json").read_bytes()).hexdigest(),
             "actual_target_hidden_activations":False,"details":{}}
        canonical=dict(agent="speculation",candidate="real_checkpoint_full_projection",
            scope="single SM86, real DFlash2 FC, hidden_norm, five TP0 K/V slices plus K-normalizations and true configured RoPE, synthetic changing upstream taps",
            baseline_sha256=BASELINE,correct=True,active=True,stage=stage,seed=seed,
            unit="synchronized wall microseconds",bench={"instances":1,"shapes":{}})
        for b in (1,4,16,32):
            for risk in ("partial","full"):
                name=f"B{b}_{risk}"
                frozen=FrozenNoSelection(fc,b*8,selections[str(b*8)]["chosen"])
                try:
                    row=case(b,seed,fc,kv,norm,rope,rms,args.rounds,frozen,risk)
                    row["frozen_first_input_mismatches"]=frozen.initial_bad
                    row["all_replays_changed"]=(len(row["changed_projection_values"])==args.rounds-1 and min(row["changed_projection_values"])>0)
                    doc["details"][name]=row
                    canonical["bench"]["shapes"][name]={"weight":1,"times":row["wall_us"]}
                    canonical["correct"]&=frozen.initial_bad==0 and row["all_replays_changed"]
                    canonical["correct"]&=all(v==0 for v in row["wrong"].values()) and all(v==0 for stages in row["wrong_stages"].values() for v in stages.values())
                    canonical["active"]&=frozen.index is not None
                    print(stage,name, "all_changed",row["all_replays_changed"],"correct",canonical["correct"],
                        {arm:round(statistics.median(values),1) for arm,values in row["wall_us"].items()},flush=True)
                    (HERE/f"real_{stage}_detail.json").write_text(json.dumps(doc,indent=2)+"\n")
                    (HERE/f"real_{stage}.json").write_text(json.dumps(canonical,indent=2)+"\n")
                finally: frozen.close()
                torch.cuda.empty_cache()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("cpu","pilot","holdout","both"))
    p.add_argument("--rounds",type=int,default=12)
    with torch.inference_mode(): main(p.parse_args())

