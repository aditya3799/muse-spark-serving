"""Compare full two-projection expert outputs with the measured serving parent on identical inputs."""
import argparse
import ast
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
from bench_experts import ROOT,OLD,SOURCE,module

HERE=Path(__file__).resolve().parent
PACKAGE=HERE/"native_package_02"

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--weight-seed",type=int,default=4318)
    args=parser.parse_args()
    out=args.out.resolve()
    pkg=(PACKAGE/"candidate.diff").read_bytes()
    assert hashlib.sha256(pkg).hexdigest()=="1ce48266f47574c5c0d4fd3c557b12f6d24811d03121f9bd9ea4e89965f6065c"
    src=(PACKAGE/"hopper_expert.cu").read_text()
    prefix=src[:src.index("constexpr int UNIT_BYTES")]
    gather=src[src.index("__device__ __forceinline__ uint4 wgs_kperm3_load"):src.index("// Work left from the previous segment")]
    epi=src[src.index("template <bool PB>\n__device__ __forceinline__ void wgs_epi_tile"):src.index("template <bool PB>\n__device__ __forceinline__ void wgs_flush")]
    cu=out/"paired.cu"
    cu.write_text(prefix+gather+epi+(HERE/"paired_mma.cuh").read_text())
    so=out/"paired.so"
    subprocess.run(["nvcc","-std=c++17","-O3","-arch=sm_86","--shared","-Xcompiler=-fPIC","--cudart=shared",str(cu),"-o",str(so)],check=True)
    run=ctypes.CDLL(str(so)).paired
    run.argtypes=[ctypes.c_void_p]*9+[ctypes.c_int]*2+[ctypes.c_void_p]
    for path in (OLD/"humming_vendor",OLD/"phase2",ROOT/"optimization_marlin_tiles_2026-11-15"):
        sys.path.insert(0,str(path))
    from kernel_runtime import bootstrap
    bootstrap(ROOT/"sglang")
    from fixture_tuning_cache import cached_threshold_for_preparation
    from native_package_02.hopper_pack import pack_experts
    moe=SOURCE/"kernels/ops/moe"
    module("sglang.kernels.ops.moe.muse_spark_humming_scale",moe/"muse_spark_humming_scale.py")
    module("sglang.kernels.ops.moe.muse_spark_humming_split",moe/"muse_spark_humming_split.py")
    finish=module("math_finalize",moe/"muse_spark_post_expert_finalize_triton.py").fused_post_expert_norm_weighted_sum
    path=SOURCE/"srt/layers/muse_spark/muse_spark_nvfp4_moe.py"
    funcs=[n for n in ast.parse(path.read_text()).body if getattr(n,"name",None) in
           ("_prepare_humming_experts","_humming_expert_outputs_default","_humming_expert_outputs")]
    ns={"torch":torch}
    exec(compile(ast.Module(funcs,[]),str(path),"exec"),ns)
    torch.manual_seed(args.weight_seed)
    raw=torch.nn.Module()
    packed=[]
    e=32
    for name,n in (("w13",8192),("w2",4096)):
        w=torch.randint(256,(e,n,2048),device="cuda",dtype=torch.uint8)
        scale=torch.randint(40,68,(e,n,256),device="cuda",dtype=torch.uint8).view(torch.float8_e4m3fn)
        setattr(raw,name+"_weight",w)
        setattr(raw,name+"_weight_scale",scale)
        setattr(raw,name+"_weight_scale_2",torch.linspace(.125,.5,e,device="cuda").bfloat16())
        packed.append(pack_experts(w,scale,name=="w13"))
    with cached_threshold_for_preparation(out):
        parent=ns["_prepare_humming_experts"](raw,torch)
    gs=raw.w13_weight_scale_2.float()
    del raw,w,scale
    fp4=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],device="cuda")
    table=(torch.arange(256,device="cuda").byte().view(torch.float8_e4m3fn).float()[:,None]*fp4).bfloat16()
    tiles=torch.zeros(30932992,device="cuda",dtype=torch.bfloat16)
    ready=torch.zeros(608,device="cuda",dtype=torch.int32)
    norm=torch.linspace(.7,1.2,4096,device="cuda").bfloat16()
    rows=[]
    def metrics(a,b):
        d=(a.float()-b.float()).flatten()
        return dict(n=d.numel(),finite=bool(torch.isfinite(d).all()),equal=float((a==b).float().mean()),
                    max_abs=float(d.abs().max()),rel_l2=float(d.norm()/b.float().norm().clamp_min(1e-12)))
    def save():
        (out/"results.json").write_text(json.dumps(dict(device=torch.cuda.get_device_name(),weight_seed=args.weight_seed,rows=rows,
            patch=hashlib.sha256(pkg).hexdigest(),scope="SM86 warp MMA, global weight loads and three ordered kernels substitute for Hopper warpgroup MMA, TMA and cooperative scheduling. Real packed weights, gather, both epilogues and scored finalizer. No H200 timing."),indent=2))
    for m,pattern,seed in [(1,"one",3),(8,"uniform",8),(32,"hot",13),(128,"uniform",17),(128,"uniform",23),(256,"hot",31),(385,"uniform",45),(1024,"dense",144),(753,"uniform",61)]:
        torch.manual_seed(seed)
        ids=torch.rand(m,256,device="cuda").topk(8).indices.int()
        ids=torch.where(ids<32,ids,-1).contiguous()
        if pattern=="dense": ids=torch.rand(m,32,device="cuda").topk(8).indices.int().contiguous()
        ids[0,0]=0
        if pattern=="hot": ids[:,0]=0
        x=torch.randn(m,4096,device="cuda",dtype=torch.bfloat16)*.023
        scores=torch.rand(m,8,device="cuda"); scores/=scores.sum(-1,keepdim=True)
        ref,_=ns["_humming_expert_outputs"](parent,x,ids,torch,defer_downscale=True,global_experts=256)
        actual=torch.empty_like(ref)
        def project():
            for first in range(0,m,384):
                args=(*packed,gs,table,x[first],ids[first],actual[first],tiles,ready)
                err=run(*(a.data_ptr() for a in args),min(m-first,384),e,torch.cuda.current_stream().cuda_stream)
                if err: raise RuntimeError(err)
        project()
        def finalize(down):
            return finish(down,scores,norm_weight=norm,eps=1e-8,weight_first=False,slot_expert_ids=ids,expert_output_scale=parent.w2_post_scale)
        expected=finalize(ref)
        result=finalize(actual)
        raw_result=metrics(actual[ids>=0],ref[ids>=0])
        final=metrics(result,expected)
        row=dict(m=m,pattern=pattern,seed=seed,routes=int((ids>=0).sum()),raw=raw_result,final=final,active_final=metrics(result[(ids>=0).any(-1)],expected[(ids>=0).any(-1)]))
        rows.append(row)
        save()
        assert final["finite"] and raw_result["finite"] and final["rel_l2"]<.02,(m,row)
        if m==753:
            packed[1].view(32,-1)[0].zero_()
            project()
            changed=metrics(finalize(actual),result)
            row["corrupted_w2"]=changed
            save()
            assert changed["rel_l2"]>.01,changed
if __name__=="__main__":
    main()

