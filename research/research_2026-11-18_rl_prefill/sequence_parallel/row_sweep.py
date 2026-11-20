"""Screen parent/native-candidate cuBLASLt plans for the exact split-router FP32 partials."""
import ctypes as ct
import json
import sys
import statistics
import time
from pathlib import Path
import torch
from torch import nn
from router_gpu import source, HERE, ROOT

sys.path.insert(0,str(ROOT/"research_2026-11-18_lt_audit"))
from runtime_projection import native_workspace

def load():
    lib=ct.CDLL(str(HERE/"row_sweep.so"))
    ptr=ct.c_void_p;I=ct.c_int;U=ct.c_uint64
    for name,args,ret in (("make_sweep",[I]*5+[U,U,I],ptr),("sweep_count",[ptr],I),
                         ("sweep_call",[ptr,I,I,ptr,ptr,ptr,ptr,U,ptr],I),
                         ("sweep_attr",[ptr,I,ct.POINTER(I),ct.POINTER(U)],None),("drop_sweep",[ptr],None)):
        fn=getattr(lib,name);fn.argtypes=args;fn.restype=ret
    return lib

def bits(a,b):return torch.equal(a.view(torch.int32),b.view(torch.int32))
def main():
    import argparse,faulthandler
    parser=argparse.ArgumentParser()
    parser.add_argument("--repeat-only",action="store_true");parser.add_argument("--rows",nargs="+",type=int,default=[512,520,16384])
    a=parser.parse_args()
    faulthandler.dump_traceback_later(30,repeat=True)
    print("torch runtime loaded",flush=True)
    lib=load();native=native_workspace();assert native
    print("workspace query",native,flush=True)
    cap=max(native[1],32*1024**2)
    scratch=torch.empty(cap,dtype=torch.uint8,device="cuda")
    klass=source(HERE/"muse_spark.py",{"torch":torch,"nn":nn},["MuseSpark1xRouter"]).MuseSpark1xRouter
    report={"native_workspace":native,"arch":torch.cuda.get_device_name(),"rows":{},"heldout_seeds":[229819,772141]}
    for m in a.rows:
        local=m//8
        result=report["rows"][m]={"repeat_checks":[],"candidates":[]}
        t0=time.monotonic()
        print("vendor discovery",m,"repeat-only",a.repeat_only,flush=True)
        q=None if a.repeat_only else lib.make_sweep(local,512,m,8192,512,native[1],cap,-1)
        count=lib.sweep_count(q) if q else 0
        result["count"]=count
        print("plans",count,flush=True)
        def call(i,x,w,full=False):
            dst=torch.empty(x.shape[0],512,device="cuda")
            rc=lib.sweep_call(q,i,int(full),w.data_ptr(),x.data_ptr(),dst.data_ptr(),scratch.data_ptr(),cap,torch.cuda.current_stream().cuda_stream)
            assert rc==0,rc
            return dst
        def metadata(i):
            at=(ct.c_int*10)();ws=(ct.c_uint64*3)();lib.sweep_attr(q,i,at,ws)
            return {"index":i,"fields":list(at),"workspace":list(ws)}
        candidates=list(range(count))
        for seed,style in ((62813,"random"),(62817,"near_tie"),(62819,"cancel"),(229819,"random"),(772141,"near_tie")):
            print("fixture",m,seed,style,flush=True)
            gen=torch.Generator(device="cuda").manual_seed(seed)
            x=torch.randn(m,8192,device="cuda",generator=gen,dtype=torch.bfloat16)
            w=torch.randn(256,8192,device="cuda",generator=gen)/8192**.5
            if style=="near_tie":w[1::2]=w[::2]+torch.randn(128,8192,generator=gen,device="cuda")*2e-10
            if style=="cancel":
                w[:,1::2]=-w[:,::2];x[:,1::2]=x[:,::2];x[:,-1]+=0.00390625
            router=klass(8192,256,torch.float32).cuda();router.weight.copy_(w);router._build_split2()
            packed=router._split2_weight
            ref=torch.mm(x,packed,out_dtype=torch.float32)
            # Exercise rank zero, a central shard, and the final shard on every seed.
            for rank in (range(8) if seed in report['heldout_seeds'] else (0,3,7)):
                sl=slice(rank*local,(rank+1)*local)
                part=x[sl]; expected=ref[sl]
                # Every retained FP32 result has the same original row address and stride.
                padded=part.repeat((8,1))
                repeated=torch.mm(padded,packed,out_dtype=torch.float32)[sl]
                result["repeat_checks"].append([seed,style,rank,bits(repeated,expected)])
                if candidates:
                    candidates=[i for i in candidates if bits(call(i,part,packed),expected)]
            print("routes compared",m,seed,len(candidates),flush=True)
            if seed==62813:
                result["exact_local_after_first_random"]=len(candidates)
                candidates=[i for i in candidates if bits(call(i,x,packed,True),ref)]
                result["also_exact_full_anchor"]=len(candidates)
            result["candidates"].append([seed,style,len(candidates)])
        result["passing"]=[metadata(i) for i in candidates]
        result["elapsed_s"]=round(time.monotonic()-t0,2)
        result["replicated_full_bytes"]=m*8192*2
        if q: lib.drop_sweep(q)
        (HERE/("row_repeat.json" if a.repeat_only else "row_sweep.json")).write_text(json.dumps(report,indent=2))
        print(m,count,result["candidates"],all(a[-1] for a in result["repeat_checks"]),flush=True)

if __name__=="__main__":main()

