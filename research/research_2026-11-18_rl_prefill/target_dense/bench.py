"""Time a complete target pooler or unpooler GEMM as dispatched by the scored H200 TP8 model."""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
AUDIT=ROOT/"research_2026-11-18_lt_audit"
sys.path.insert(0,str(AUDIT))
BASELINE="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"

def dimensions(rows, projection):
    n,k=(4096,8192) if projection=="pool" else (8192,4096)
    if rows<=0: raise ValueError("Expected positive token count")
    if rows<=8 and projection=="unpool":
        return "tiny_columns",rows,n//8,k
    if rows<512:
        return "columns",rows,n//8,k
    if rows%8==0:
        return "rows",rows//8,n,k
    return "replicated",rows,n,k

def structural():
    import ast
    assert hashlib.sha256((ROOT/"release_2026-11-18/candidate.diff").read_bytes()).hexdigest()==BASELINE
    p=ROOT/"release_2026-11-18/candidate_sources/python/sglang/srt/layers/communication/muse_spark_gemm_ag.py"
    tree=ast.parse(p.read_bytes())
    expected={"small_projection_local":"torch.mm(x, weight[start:start + n_local].t(), out_dtype=torch.float32)",
              "prefill_row_projection":"torch.mm(x[start:start + m_local], weight.t(), out_dtype=torch.float32)"}
    for f in tree.body:
        if getattr(f,"name",None) in expected:
            assert expected[f.name] in ast.unparse(f)
            assert ".to(x.dtype)" in ast.unparse(f)
    assert set(expected).issubset({getattr(f,"name",None) for f in tree.body})
    assert dimensions(16384,"pool")==("rows",2048,4096,8192)
    assert dimensions(753,"unpool")==("replicated",753,8192,4096)
    assert dimensions(128,"pool")==("columns",128,512,8192)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--rows",type=int,nargs="+",default=[128,256,753,16384])
    parser.add_argument("--projections",nargs="+",choices=["pool","unpool"],default=["pool","unpool"])
    parser.add_argument("--out",type=Path)
    parser.add_argument("--pairs",type=int,default=12)
    parser.add_argument("--max-split",type=int,default=4)
    args=parser.parse_args()
    structural()
    if args.preflight:
        print("Scored shard graph and output dtype verified")
        return
    import torch
    import torch.nn.functional as F
    from audit_match import load_lib,describe
    from runtime_projection import native_workspace
    assert args.out and args.pairs>=4 and args.pairs%2==0
    args.out.mkdir(exist_ok=True,parents=True)
    libs={"bf16":load_lib()}
    libs["fp32"]=ct.CDLL(str(HERE/"sweep_fp32.so"))
    for name in ("make_sweep","sweep_count","sweep_attr","sweep_call","drop_sweep"):
        source,target=getattr(libs["bf16"],name),getattr(libs["fp32"],name)
        target.argtypes,target.restype=source.argtypes,source.restype
    native=native_workspace()
    assert native is not None
    cap=max(native[1],32*1024**2)
    scratch=torch.empty(cap,device="cuda",dtype=torch.uint8)
    seeds={"pilot":218693,"holdout":782411}
    results={key:dict(agent="prefill",candidate="target_dense_vendor",scope="rank-local full-K target pooler/unpooler with scored output type; no collective",
                      baseline_sha256=BASELINE,seed=seed,correct=True,active=False,complete=False,bench=dict(instances=1,shapes={})) for key,seed in seeds.items()}
    detail=dict(device=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),native_workspace=native,device_bytes=cap,shapes={})
    def save():
        for key,record in [*results.items(),("detail",detail)]:
            (args.out/(key+".json")).write_text(json.dumps(record,indent=2)+"\n")
    save()
    def timing(fn):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();fn();b.record();b.synchronize()
        return a.elapsed_time(b)*1000
    for rows in args.rows:
        for proj in args.projections:
            kind,m,n,k=dimensions(rows,proj)
            key=f"{proj}_M{rows}_{kind}"
            dtype=torch.bfloat16 if kind in ("replicated","tiny_columns") else torch.float32
            lib=libs["bf16" if kind=="replicated" else "fp32"]
            q=lib.make_sweep(m,n,n,k,n,native[1],cap,args.max_split)
            assert q
            def inputs(seed,case):
                gen=torch.Generator(device="cuda").manual_seed(seed)
                x=torch.empty(m,k,device="cuda",dtype=torch.bfloat16).normal_(generator=gen)
                w=torch.empty(n,k,device="cuda",dtype=torch.bfloat16).normal_(std=.012,generator=gen)
                if case=="cancel":
                    xs,ws=x.view(m,-1,2,256),w.view(n,-1,2,256)
                    xs[:,:,1].copy_(xs[:,:,0]);ws[:,:,1].copy_(ws[:,:,0]).neg_()
                    ws[::3,:,1,0].mul_(1.0078125)
                if case=="finite":
                    # Alternate magnitudes and signs by K group, including signed zeros.
                    x[:,::4]*=128
                    w[:,::4]/=128
                    x[::11,1::7]=-0.0
                    w[::13,::17]=0
                return x,w
            def baseline_raw(x,w):
                if dtype==torch.bfloat16: return F.linear(x,w)
                return torch.mm(x,w.t(),out_dtype=torch.float32)
            def baseline(x,w):
                return baseline_raw(x,w).to(torch.bfloat16)
            def candidate_raw(i,x,w):
                y=torch.empty(m,n,device=x.device,dtype=dtype)
                status=lib.sweep_call(q,i,1,w.data_ptr(),x.data_ptr(),y.data_ptr(),scratch.data_ptr(),cap,torch.cuda.current_stream().cuda_stream)
                if status: raise RuntimeError(f"cuBLASLt status {status}")
                return y
            def candidate(i,x,w):
                return candidate_raw(i,x,w).to(torch.bfloat16)
            def mismatches(y,ref):
                storage=torch.int16 if ref.dtype==torch.bfloat16 else torch.int32
                return int(torch.count_nonzero(y.view(storage)!=ref.view(storage)))
            examples=[]
            for idx,case in enumerate(("random","cancel","finite")):
                x,w=inputs(17913+idx,case)
                raw=baseline_raw(x,w)
                assert torch.isfinite(raw).all()
                examples.append((x,w,raw))
            count=lib.sweep_count(q)
            assert count>0, "No vendor configurations for this shape"
            receipt=dict(native_output=str(dtype),m=m,n=n,k=k,eligible=count,passing=[],selection=None)
            detail["shapes"][key]=receipt
            passing=[]
            for i in range(count):
                exact=True
                for x,w,ref in examples:
                    try: y=candidate_raw(i,x,w)
                    except RuntimeError:
                        exact=False
                        break
                    # An FP32 intermediate must pass at FP32 precision before its BF16 cast.
                    if mismatches(y,ref):
                        exact=False
                        break
                if exact:
                    us=[timing(lambda:candidate(i,x,w)) for _ in range(3)]
                    passing.append((statistics.median(us),i))
                    receipt["passing"].append(dict(index=i,**describe(lib,q,i),screen_us=us))
            i=min(passing)[1] if passing else None
            receipt["selection"]=None if i is None else dict(index=i,**describe(lib,q,i))
            del examples,x,w,raw,ref
            print(key,count,"candidates",len(passing),"exact",receipt["selection"],flush=True)
            for stage,seed in seeds.items():
                x,w=inputs(seed,"random")
                funcs={"baseline":lambda:baseline(x,w),"control":lambda:baseline(x,w),
                       "candidate":(lambda:baseline(x,w)) if i is None else (lambda:candidate(i,x,w))}
                graph,outputs={},{}
                for arm,fn in funcs.items():
                    fn();torch.cuda.synchronize()
                    g=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g): outputs[arm]=fn()
                    graph[arm]=g
                for _ in range(2):
                    for g in graph.values():g.replay()
                r=dict(weight=1,times={arm:[] for arm in funcs},order=[],active=i is not None,
                       max_intermediate_mismatches=0,max_return_mismatches=0,max_control_mismatches=0)
                results[stage]["bench"]["shapes"][key]=r
                for step in range(args.pairs):
                    a,b=inputs(seed+step*43,("random","cancel","finite")[step%3])
                    x.copy_(a);w.copy_(b)
                    del a,b
                    if i is not None:
                        ref=baseline_raw(x,w)
                        bad=mismatches(candidate_raw(i,x,w),ref)
                        r["max_intermediate_mismatches"]=max(r["max_intermediate_mismatches"],bad)
                        del ref
                    order=["baseline","candidate","control"] if step%2==0 else ["control","candidate","baseline"]
                    r["order"].append(order)
                    for arm in order:r["times"][arm].append(timing(graph[arm].replay))
                    bad=mismatches(outputs["baseline"],outputs["candidate"])
                    bad_copy=mismatches(outputs["baseline"],outputs["control"])
                    r["max_return_mismatches"]=max(bad,r["max_return_mismatches"])
                    r["max_control_mismatches"]=max(bad_copy,r["max_control_mismatches"])
                results[stage]["active"] |= i is not None
                results[stage]["correct"] &= max(r["max_intermediate_mismatches"],r["max_return_mismatches"],r["max_control_mismatches"])==0
                print(stage,key,{a:round(statistics.median(t),1) for a,t in r["times"].items()},r["max_intermediate_mismatches"],flush=True)
                save()
                del graph,outputs,g,x,w
            lib.drop_sweep(q)
            torch.cuda.empty_cache()
    for d in results.values(): d["complete"]=True
    save()

if __name__=="__main__": main()

