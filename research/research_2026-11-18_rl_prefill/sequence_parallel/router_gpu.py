"""Replay the scored full-width split-router and pinned Triton top-8 on changed row shards."""
import ast
import hashlib
import json
from pathlib import Path
import statistics
import sys
import types
from typing import Optional
import torch
from torch import nn
import triton
import triton.language as tl
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def source(path,ns,nodes):
    tree=ast.parse(path.read_bytes())
    keep=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in nodes]
    assert set(nodes)=={n.name for n in keep}
    module=types.ModuleType("candidate_"+path.stem)
    module.__file__=str(path); module.__dict__.update(ns); sys.modules[module.__name__]=module
    exec(compile(ast.Module(ast.parse("from __future__ import annotations").body+keep,[]),str(path),"exec"),module.__dict__)
    return module

def abba(do,small,big):
    out={"baseline":[],"candidate":[],"control":[]}
    for i in range(8):
        for mode in (("baseline","candidate","control") if i%2==0 else ("control","candidate","baseline")):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record()
            do(small if mode=="candidate" else big)
            end.record();end.synchronize()
            out[mode].append(start.elapsed_time(end)*1000)
    return out

def main():
    path=ROOT/"sglang/python/sglang/kernels/ops/moe/muse_spark_routing.py"
    names=[f.name for f in ast.parse(path.read_bytes()).body if isinstance(f,ast.FunctionDef) and f.name!="fused_muse_spark_routing"]
    ops=source(path,{"torch":torch,"triton":triton,"tl":tl,"Optional":Optional,"register_custom_op":lambda **kw:lambda f:f},names)
    model=source(HERE/"muse_spark.py",{"torch":torch,"nn":nn},["MuseSpark1xRouter"])
    rows_list=(512,520,16384)
    report={"device":torch.cuda.get_device_name(),"torch":str(torch.__version__),"source":hashlib.sha256(path.read_bytes()).hexdigest(),"cases":[]}
    for seed in (723041,885127):
        g=torch.Generator(device="cuda").manual_seed(seed)
        router=model.MuseSpark1xRouter(8192,256,torch.float32).cuda()
        for style in ("random","near_tie","cancellation"):
            w=torch.randn(256,8192,generator=g,device="cuda")/8192**.5
            if style=="near_tie":
                w[1::2]=w[::2]+torch.randn(128,8192,generator=g,device="cuda")*2e-10
            if style=="cancellation":
                w[:,1::2]=-w[:,::2]
            router.weight.copy_(w)
            bias=torch.zeros(256,device="cuda")
            def run(x,with_logits=False):
                logits=router.forward_split2(x)
                shape=(x.shape[0],8)
                weights=torch.empty(shape,device="cuda");ids=torch.empty(shape,device="cuda",dtype=torch.int32)
                packed=torch.empty_like(ids)
                ops.fused_muse_spark_routing_op(logits,bias,weights,ids,packed,8,1.0,1.0,1e-15,0,False,True)
                return (logits,weights,ids,packed) if with_logits else (weights,ids,packed)
            for rows in rows_list:
                x=torch.randn(rows,8192,generator=g,device="cuda",dtype=torch.bfloat16)
                if style=="cancellation":
                    x[:,1::2]=x[:,::2]
                    x[:,-1]+=0.00390625
                base=run(x,True)
                split=[run(a,True) for a in x.chunk(8,dim=0)]
                trial=[torch.cat([a[j] for a in split]) for j in range(4)]
                report["cases"].append(c:=dict(seed=seed,fixture=style,rows=rows,
                    diffs={key:int((a!=b).sum()) for key,a,b in zip(("logits","weights","ids","packed"),base,trial)},
                    max_abs_logit_error=float((base[0]-trial[0]).abs().max()),
                    rows_with_different_top8=int((base[2]!=trial[2]).any(1).sum())))
                if style=="random":
                    c["rank_local_times_us"]=abba(run,x[:rows//8],x)
                del base,trial,split
                print(seed,style,rows,c["diffs"],flush=True)
            del w
    (HERE/"router_gpu.json").write_text(json.dumps(report,indent=2))
if __name__=="__main__": main()

