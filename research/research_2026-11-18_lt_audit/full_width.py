"""Select native full-width cuBLASLt matmul at runtime with portable matching and honest A/B."""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from audit_match import BASELINE, N, K, load_lib, describe, mismatches
from runtime_projection import native_workspace

PILOT, HOLDOUT=915841,224737
HERE=Path(__file__).resolve().parent


def data(m,seed,stress=False):
    g=torch.Generator(device="cuda").manual_seed(seed)
    x=torch.empty(m,K,device="cuda",dtype=torch.bfloat16).normal_(generator=g)
    w=torch.empty(N,K,device="cuda",dtype=torch.bfloat16).normal_(std=.006,generator=g)
    if stress:
        xs,ws=x.view(m,-1,2,256),w.view(N,-1,2,256)
        xs[:,:,1,:].copy_(xs[:,:,0,:])
        ws[:,:,1,:].copy_(ws[:,:,0,:]).neg_()
        ws[::3,:,1,0].mul_(1.0078125)
    return x,w,F.linear(x,w)


def timing(fn):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record()
    out=fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b)*1000


def paired(x,w,fn,pairs,seed):
    g=torch.Generator(device="cuda").manual_seed(seed)
    xs=[x]+[torch.empty_like(x).normal_(generator=g) for _ in range(3)]
    incorrect=[mismatches(fn(v,w),F.linear(v,w)) for v in xs]
    rr=random.Random(seed)
    arms=["baseline","candidate","control"]
    times={arm:[] for arm in arms}
    orders=[]
    for i in range(pairs):
        order=arms.copy()
        rr.shuffle(order)
        value=xs[i%len(xs)]
        for o in (order,list(reversed(order))):
            orders.append(o)
            for arm in o:
                call=fn if arm=="candidate" else F.linear
                times[arm].append(timing(lambda:call(value,w)))
    return times,orders,incorrect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows",nargs="+",type=int,default=[8,32,128,256,753])
    parser.add_argument("--max-split",type=int,default=8)
    parser.add_argument("--pairs",type=int,default=6)
    parser.add_argument("--prefix",default="full_width")
    args=parser.parse_args()
    lib=load_lib()
    native=native_workspace()
    assert native is not None
    cap=max(native[1],32*2**20)
    scratch=torch.empty(cap,device="cuda",dtype=torch.uint8)
    stream=torch.cuda.current_stream().cuda_stream
    results={s:dict(agent="speculation",candidate="full_width_vendor_matmul",scope="one physical SM86 native BF16 full-width FC, no TP",
        baseline_sha256=BASELINE,seed=s,correct=True,active=True,bench={"instances":1,"shapes":{}})
        for s in (PILOT,HOLDOUT)}
    details={}
    def save():
        for stage,seed in (("pilot",PILOT),("holdout",HOLDOUT)):
            (HERE/f"{args.prefix}_{stage}.json").write_text(json.dumps(results[seed],indent=2)+"\n")
        (HERE/f"{args.prefix}_detail.json").write_text(json.dumps(details,indent=2)+"\n")
    for m in args.rows:
        x,w,ref=data(m,PILOT)
        stress_x,stress_w,stress_ref=data(m,PILOT+1,True)
        torch.cuda.synchronize()
        start=time.perf_counter()
        q=lib.make_sweep(m,N,N,K,N,native[1],cap,args.max_split)
        if not q: raise RuntimeError("vendor search unavailable")
        count=lib.sweep_count(q)
        receipt={"eligible":count,"native_workspace":native[1],"search_workspace_limit":cap,
                 "max_split":args.max_split,"results":[],"gpu":str(torch.cuda.get_device_properties(0))}
        details[m]=receipt
        def call(i,inp,weight):
            out=torch.empty((m,N),device=inp.device,dtype=inp.dtype)
            status=lib.sweep_call(q,i,1,weight.data_ptr(),inp.data_ptr(),out.data_ptr(),scratch.data_ptr(),cap,stream)
            if status: return None
            return out
        print("full-width",m,"eligible",count,flush=True)
        passing=[]
        for i in range(count):
            out=call(i,x,w)
            if out is None: continue
            bad=mismatches(out,ref)
            if bad: continue
            alternative=call(i,stress_x,stress_w)
            if alternative is None or mismatches(alternative,stress_ref): continue
            measures=[timing(lambda:call(i,x,w)) for _ in range(3)]
            passing.append((statistics.median(measures),i))
            receipt["results"].append({"index":i,**describe(lib,q,i),"screen_us":measures})
        del stress_x,stress_w,stress_ref
        receipt["startup_ms"]=1000*(time.perf_counter()-start)
        chosen=min(passing)[1] if passing else None
        receipt["chosen"]=None if chosen is None else {"index":chosen,**describe(lib,q,chosen)}
        fn=F.linear if chosen is None else lambda v,wt:call(chosen,v,wt)
        for seed in (PILOT,HOLDOUT):
            if seed==HOLDOUT:
                del x,w,ref
                x,w,ref=data(m,HOLDOUT)
            times,orders,incorrect=paired(x,w,fn,args.pairs,seed)
            sx,sw,sr=data(m,seed+1,True)
            error=mismatches(fn(sx,sw),sr)
            del sx,sw,sr
            result=results[seed]
            result["active"]&=chosen is not None
            result["correct"]&=(max(incorrect)==0 and error==0)
            name=f"M{m}"
            result["bench"]["shapes"][name]={"weight":1,"times":times}
            receipt[str(seed)]={"orders":orders,"fresh_mismatches":incorrect,"cancellation_mismatches":error}
        del x,w,ref
        lib.drop_sweep(q)
        print("full-width",m,"pass",len(passing),"winner",chosen,
            "holdout medians",{k:round(statistics.median(v),1) for k,v in times.items()},flush=True)
        save()

if __name__=="__main__": main()


