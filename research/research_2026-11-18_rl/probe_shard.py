"""Test unmodified full-K cuBLASLt algorithms against the exact full-width native BF16 output."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import statistics

import torch
from build_aux import DIGEST, PARENT

HERE = Path(__file__).resolve().parent


def native():
    lib=ct.CDLL(str(HERE/"lt_probe.so"))
    ptr,num,size=ct.c_void_p,ct.c_int,ct.c_uint64
    for name,signature,returns in [
        ("prepare_probe",[num,num,num,num,size],ptr),("probe_count",[ptr],num),
        ("probe_call",[ptr,num,ptr,ptr,ptr,ptr,size,ptr],num),("destroy_probe",[ptr],None),
        ("probe_attr",[ptr,num,num],num)]:
        func=getattr(lib,name)
        func.restype,func.argtypes=returns,signature
    return lib


def timed(fn):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record()
    fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b)*1000.


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--selection",type=Path)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    assert hashlib.sha256((PARENT/"candidate.diff").read_bytes()).hexdigest()==DIGEST
    previous=json.loads(args.selection.read_text())["shapes"] if args.selection else None
    lib=native()
    rng=torch.Generator(device="cuda").manual_seed(args.seed)
    w=torch.empty(8192,40960,dtype=torch.bfloat16,device="cuda").normal_(std=.006,generator=rng)
    workspace=torch.empty(32*1024*1024,dtype=torch.uint8,device="cuda")
    stream=torch.cuda.current_stream().cuda_stream
    result=dict(seed=args.seed,baseline_sha256=DIGEST,gpu=torch.cuda.get_device_name(),shapes={},
                scope="full 8192 outputs across 8 virtual shards on one physical GPU; virtual parallel rate excludes all communication")
    for m in [8,32,128,256,753]:
        x=torch.empty(m,40960,dtype=torch.bfloat16,device="cuda").normal_(generator=rng)
        full=torch.nn.functional.linear(x,w)
        shards=w.split(1024)
        outs=[torch.empty(m,1024,dtype=torch.bfloat16,device="cuda") for _ in shards]
        plan=lib.prepare_probe(m,1024,8192,40960,workspace.numel())
        if not plan: raise RuntimeError("cuBLASLt setup failed")
        try:
            def invoke(index,rank):
                return lib.probe_call(plan,index,shards[rank].data_ptr(),x.data_ptr(),outs[rank].data_ptr(),
                                      workspace.data_ptr(),workspace.numel(),stream)
            algorithms=[]
            matched=None
            choice=None if previous is None else previous[str(m)]["chosen_config"]
            for index in range(lib.probe_count(plan)):
                config=[lib.probe_attr(plan,index,a) for a in range(5)]
                if previous is not None and choice!=config: continue
                status=invoke(index,0)
                if status:
                    algorithms.append(dict(index=index,config=config,status=status))
                    continue
                bad=int((outs[0]!=full[:,:1024]).sum())
                row=dict(index=index,config=config,rank0_disagreements=bad)
                if bad==0:
                    wrong=0
                    for rank in range(1,8):
                        assert invoke(index,rank)==0
                        wrong+=int((outs[rank]!=full[:,rank*1024:(rank+1)*1024]).sum())
                    row["remaining_disagreements"]=wrong
                    if wrong==0:
                        matched=index
                        choice=config
                algorithms.append(row)
                if matched is not None: break
            if matched is None: choice=None
            native_wrong=sum(int((torch.nn.functional.linear(x,shards[i])!=full[:,i*1024:(i+1)*1024]).sum()) for i in range(8))
            row=dict(algorithms=algorithms,chosen_config=choice,naive_shard_disagreements=native_wrong,
                     total_outputs=full.numel(),planner_options=lib.probe_count(plan))
            if matched is not None:
                dest=torch.empty_like(full)
                def serial():
                    for rank in range(8):
                        assert invoke(matched,rank)==0
                    torch.cat(outs,dim=1,out=dest)
                def baseline():
                    torch.nn.functional.linear(x,w)
                def rank_only():
                    assert invoke(matched,0)==0
                times={k:[] for k in ["baseline","serial","rank_only","control"]}
                funcs=dict(baseline=baseline,control=baseline,serial=serial,rank_only=rank_only)
                for fn in funcs.values(): fn()
                for trial in range(12):
                    order=list(funcs) if trial%2==0 else list(reversed(funcs))
                    for name in order: times[name].append(timed(funcs[name]))
                assert torch.equal(dest,full)
                row["times_us"]=times
            result["shapes"][m]=row
            args.out.write_text(json.dumps(result,indent=2)+"\n")
            print(m,"options",lib.probe_count(plan),"match",choice,"default mismatches",native_wrong,flush=True)
        finally:
            lib.destroy_probe(plan)


if __name__=="__main__":
    main()
