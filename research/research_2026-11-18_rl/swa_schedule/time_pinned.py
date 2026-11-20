"""Measure the pinned FA3 local scheduler before or once per sliding layer on SM86."""
import argparse
import ctypes
import hashlib
import json
import math
import statistics
from pathlib import Path
import torch

here=Path(__file__).resolve().parent
root=here.parents[1]
def ceildiv(a,b):return (a+b-1)//b
def choose(b,q,bm,bn,cap,sms,window):
    mn=ceildiv(b*q*8,bm)*2
    nb=ceildiv(min(cap,window+1+bm),bn)
    if mn>=.8*sms or nb<=4:return 1
    eff=[(mn*s/sms)/math.ceil(mn*s/sms) for s in range(1,min(128,sms,nb)+1)]
    return next(i+1 for i,v in enumerate(eff) if v>=.85*max(eff))

def expected(lens,q,bm,bn,splits,sms):
    m=[ceildiv(q*8,bm)]*len(lens)
    n=[ceildiv(v,bn) for v in lens]
    # The pinned CUDA kernel uses float32 for the 10% scheduling margin.
    count=torch.tensor(sum(x*y for x,y in zip(m,n)),dtype=torch.float32)
    budget=max(1,math.ceil(float(count*1.1*2/sms)))
    d=[max(1,min(ceildiv(x,budget),splits)) if splits>1 else 1 for x in n]
    n=[max(1,ceildiv(x,s)) for x,s in zip(n,d)]
    l2=(8*1024*1024)//(bn*128)
    heads=[min(2,next((v for v in (16,8,4,2) if x*v<=l2),1)) for x in n]
    return d+m+heads

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--seed",type=int,required=True)
    args=parser.parse_args();rng=torch.Generator(device="cuda").manual_seed(args.seed)
    lib=ctypes.CDLL(str(here/"scheduler_native.so"))
    ptr=ctypes.c_void_p;integer=ctypes.c_int
    lib.okc_plan.argtypes=[ptr,ptr,ptr]+[integer]*7+[ptr]
    lib.okc_observe.argtypes=[ptr,ptr,integer,ptr]
    for op in (lib.okc_plan,lib.okc_observe):op.restype=integer
    scopes="Actual pinned FA3 CUDA variable-length local-window scheduler body compiled for SM86. Compare rebuilding it per sliding layer with a graph-owned schedule per forward and the same 46 device consumers. Includes a proxy metadata reader, plus a semaphore clear for the supplied-plan single-split case. Hopper-style tile and split inputs on local hardware. Native FP8 mainloop, PDL, TP8 and the rest of the server are excluded."
    out=dict(agent="root",candidate="Reuse one local-window FA3 schedule for 46 layers, native CUDA scheduler microbenchmark",
        baseline_sha256=hashlib.sha256((root/"release_2026-11-18/candidate.diff").read_bytes()).hexdigest(),
        candidate_sha256=json.loads((here/"compile.json").read_text())["kernel_sha256"],active=True,scope=scopes,seed=args.seed,
        complete=False,correct=True,bench=dict(instances=1,shapes={}))
    path=here/f"pinned_{args.seed}.json"
    def save():path.write_text(json.dumps(out,indent=2))
    save()
    for b,q in ((1,8),(4,8),(16,8),(32,8)):
        for bm,bn in ((64,128),(192,160)):
            cap=131072;sms=132;w=2047;rounded=ceildiv(b,4)*4
            splits=choose(b,q,bm,bn,cap,sms,w)
            lens=torch.full((b,),4096,device="cuda",dtype=torch.int32)
            cu=torch.arange(b+1,dtype=torch.int32,device="cuda")*q
            states={nm:(torch.empty(3*rounded+1,dtype=torch.int32,device="cuda"),
                        torch.empty((46,3*b),dtype=torch.int32,device="cuda")) for nm in ("baseline","candidate","control")}
            def run(name):
                workspace,audit=states[name]
                stream=torch.cuda.current_stream().cuda_stream
                for layer in range(46):
                    if name!="candidate" or layer==0:
                        assert lib.okc_plan(lens.data_ptr(),cu.data_ptr(),workspace.data_ptr(),b,q,cap,bm,bn,sms,splits,stream)==0
                    assert lib.okc_observe(workspace.data_ptr(),audit[layer].data_ptr(),b,stream)==0
                    if splits==1 and name=="candidate":workspace[-1:].zero_()
            graphs={}
            for nm in states:
                run(nm)
                with torch.cuda.graph(g:=torch.cuda.CUDAGraph()):run(nm)
                graphs[nm]=g
            times={nm:[] for nm in states}
            validations=0
            for i in range(12):
                lens.random_(1,74160,generator=rng)
                if b>1 and i%3==0:lens[-1:]=0
                gold=torch.tensor(expected(lens.tolist(),q,bm,bn,splits,sms),device="cuda",dtype=torch.int32)
                sequence=("baseline","candidate","control") if i%2==0 else ("control","candidate","baseline")
                for nm in sequence:
                    t0=torch.cuda.Event(enable_timing=True);t1=torch.cuda.Event(enable_timing=True)
                    t0.record()
                    for _ in range(10):graphs[nm].replay()
                    t1.record();t1.synchronize()
                    times[nm].append(t0.elapsed_time(t1)*100.)
                    ws,used=states[nm]
                    assert torch.equal(used,gold[None,:].expand_as(used)) and int(ws[-1])==0,(nm,b,q,bm,bn,i,used[0].tolist(),gold.tolist())
                validations+=46*3*b
            key=f"b{b}_q{q}_tile{bm}x{bn}"
            out["bench"]["shapes"][key]=dict(weight=1,static_splits=splits,times=times,checked_ints_per_arm=validations)
            print(key,splits,{k:round(statistics.median(v),2) for k,v in times.items()},flush=True);save()
    out["complete"]=True;save()
if __name__=="__main__":main()
