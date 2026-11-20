"""Compare runtime-selected output shards with current native BF16 projection bit patterns."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "release_2026-11-18" / "candidate.diff"
BASELINE = "8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
from native_abi import load_lib, describe
M, N, K, SHARD = 256, 8192, 40960, 1024
TRAIN, HOLDOUT = 381901, 745013


def example(seed, stress=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.empty(N,K,device="cuda",dtype=torch.bfloat16).normal_(std=.006,generator=gen)
    x = torch.empty(M,K,device="cuda",dtype=torch.bfloat16).normal_(generator=gen)
    if stress:
        # Mirrored, cancelling 256-element K sections; sparse one-bit perturbation.
        x = x.view(M,-1,2,256)
        x[:,:,1,:].copy_(x[:,:,0,:])
        w = w.view(N,-1,2,256)
        w[:,:,1,:].copy_(w[:,:,0,:]).neg_()
        w[::3,:,1,0].mul_(1.0078125)
        x, w = x.view(M,K), w.view(N,K)
    return x, w, torch.nn.functional.linear(x,w)


def mismatches(a,b):
    return int(torch.count_nonzero(a.view(torch.int16) != b.view(torch.int16)))


def sweep(args):
    lib = load_lib()
    dev = torch.cuda.get_device_properties(0)
    workspace = torch.empty(args.cap_mib*2**20,dtype=torch.uint8,device="cuda")
    assumptions = {"torch": str(torch.__version__), "torch_git":torch.version.git_version,
                   "cuda": torch.version.cuda, "device":str(dev), "sm_count":dev.multi_processor_count,
                   "bf16_reduced":torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                   "workspace_env":os.getenv("CUBLAS_WORKSPACE_CONFIG"),
                   "lt_workspace_env":os.getenv("CUBLASLT_WORKSPACE_SIZE"),
                   "native_workspace_hypothesis":args.native_workspace, "max_workspace":workspace.numel()}
    doc = dict(scope="one physical GPU, eight serial slices, bit-pattern tests only",
               original_patch_sha256=BASELINE, seed=TRAIN, holdout_seed=HOLDOUT, environment=assumptions, layouts=[])
    stream = torch.cuda.current_stream().cuda_stream
    save = lambda: args.out.write_text(json.dumps(doc,indent=2)+"\n")
    x, w, ref = example(TRAIN)
    full_output = torch.empty_like(ref)
    initial_bad = sum(mismatches(torch.nn.functional.linear(x,part), ref[:,i*SHARD:(i+1)*SHARD])
                      for i,part in enumerate(w.split(SHARD)))
    doc["stock_shard_mismatches"] = initial_bad
    for ldout in args.ldout:
        local = torch.empty_strided((M,SHARD),(ldout,1),dtype=x.dtype,device=x.device)
        plan = lib.make_sweep(M,SHARD,N,K,ldout,args.native_workspace,workspace.numel(),args.max_split)
        if not plan: raise RuntimeError("cuBLASLt plan creation failed")
        count = lib.sweep_count(plan)
        part = {"ldout":ldout,"configs_checked_by_vendor":count, "results":[], "survivors":[]}
        doc["layouts"].append(part)
        def invoke(index, full, weight, inp, out):
            return lib.sweep_call(plan,index,int(full),weight.data_ptr(),inp.data_ptr(),out.data_ptr(),
                                  workspace.data_ptr(),workspace.numel(),stream)
        def check(index, val, wt, baseline):
            status = invoke(index,True,wt,val,full_output)
            if status:
                return {"full_status":status}
            whole = mismatches(full_output,baseline)
            allshards=[]
            for j, shard in enumerate(wt.split(SHARD)):
                status = invoke(index,False,shard,val,local)
                if status: return {"local_status":status}
                allshards.append(mismatches(local,baseline[:,j*SHARD:(j+1)*SHARD]))
            return {"full_bad":whole,"shard_bad":allshards}
        try:
            print("layout",ldout,"eligible plans", count,flush=True)
            for index in range(count):
                row = {"index":index, **describe(lib,plan,index)}
                status = invoke(index,False,w[:SHARD],x,local)
                row["status"]=status
                if not status:
                    row["rank0_bad"]=mismatches(local,ref[:,:SHARD])
                    if row["rank0_bad"]==0:
                        row["discovery"] = check(index,x,w,ref)
                        if row["discovery"]=={"full_bad":0,"shard_bad":[0]*8}:
                            part["survivors"].append(index)
                part["results"].append(row)
            save()
            for stress in [False,True]:
                val, wt, baseline = example(HOLDOUT+int(stress),stress=stress)
                key = "stress" if stress else "holdout"
                for i in part["survivors"]:
                    row=part["results"][i]
                    row[key]=check(i,val,wt,baseline)
                del val,wt,baseline
            part["validated"]=[i for i in part["survivors"] if all(part["results"][i][phase]
                =={"full_bad":0,"shard_bad":[0]*8} for phase in ("holdout","stress"))]
            print("discovery",part["survivors"],"validated",part["validated"],flush=True)
            save()
        finally:
            lib.drop_sweep(plan)


def trace(out):
    x,w,ref = example(TRAIN)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False) as prof:
        result = torch.nn.functional.linear(x,w)
        torch.cuda.synchronize()
    assert mismatches(result,ref)==0
    sink = out.with_suffix(".kineto.json")
    prof.export_chrome_trace(str(sink))
    kernels = [{"name":ev["name"],"args":ev.get("args",{})} for ev in json.loads(sink.read_text())["traceEvents"]
               if ev.get("cat")=="kernel"]
    out.write_text(json.dumps(kernels,indent=2)+"\n")
    print([(k["name"],k["args"].get("grid")) for k in kernels],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("task",choices=("load","sweep","trace"))
    p.add_argument("--native-workspace",type=int,default=8*1024*1024+128*1024)
    p.add_argument("--cap-mib",type=int,default=32)
    p.add_argument("--max-split",type=int,default=32)
    p.add_argument("--ldout",type=int,nargs="+",default=[1024,8192])
    p.add_argument("--out",type=Path,default=HERE/"sweep_report.json")
    args=p.parse_args()
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==BASELINE
    if args.task=="load":
        load_lib()
        assert not torch.cuda.is_initialized()
        print("native API loads without initializing CUDA")
    elif args.task=="trace": trace(args.out)
    else: sweep(args)

