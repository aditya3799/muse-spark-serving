"""Compile the exact original and fused one-CTA row kernels without a GPU."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"]=""
here=Path(__file__).resolve().parent
root=here.parents[1]
jit=root/"sglang/python/sglang/kernels/jit"
compiler=metadata.distribution("nvidia-cuda-nvcc")
nvcc=next(Path(compiler.locate_file(f)).resolve() for f in compiler.files if str(f).endswith("/bin/nvcc"))
from tvm_ffi import libinfo
import tvm_ffi

tvmroot=Path(tvm_ffi.__file__).resolve().parent
includes=[jit/"include",jit/"csrc",tvmroot/"include",tvmroot/"3rdparty/dlpack/include",nvcc.parent.parent/"include"]
cuda_inc=next(p for p in Path(sys.prefix).rglob("cuda_runtime.h") if "target" in str(p) or "nvidia" in str(p))
includes.extend([cuda_inc.parent, cuda_inc.parent/"cccl"])
includes += [p.parent.parent for p in Path(sys.prefix).rglob("dlpack.h") if p.parent.name=="dlpack"]
includes += [p.parent.parent for p in Path(sys.prefix).rglob("cuda/std") if p.is_dir()]
result=dict(status="running",kernel_source_sha256={},compiles=[])
for name in ["all_gather_norm_residual","norm_residual"]:
    p=jit/f"csrc/muse_spark/{name}.cuh";result["kernel_source_sha256"][name]=hashlib.sha256(p.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory() as temp:
    td=Path(temp)
    for threads,prefetch in [(256,True),(512,False)]:
        b=lambda x:str(x).lower()
        # Model FFN variant with post-norm learned weights and channel gates.
        source='#include "muse_spark/all_gather_norm_residual.cuh"\nusing namespace sglang;\n'
        source+=f'template __global__ void sglang::muse_spark_norm_residual_kernel<8192,{threads},true,{b(prefetch)},true,false,true,true>(const __grid_constant__ NormResidualParams);\n'
        source+=f'template __global__ void sglang::muse_spark_all_gather_norm_residual_row_kernel<8192,8,{threads},true,{b(prefetch)},true,true,true>(const __grid_constant__ muse_ag_norm::Params);\n'
        src=td/"kernels.cu";src.write_text(source)
        out=td/"kernels.cubin"
        cmd=[str(nvcc),"-arch=sm_90a","-O3","-std=c++20","-DNDEBUG","-DSGL_CUDA_ARCH=900","--expt-relaxed-constexpr","--expt-extended-lambda","--cubin","-Xptxas=-v",*(f"-I{p}" for p in includes),str(src),"-o",str(out)]
        run=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        record=dict(threads=threads,prefetch=prefetch,exit=run.returncode,message=run.stdout[-3200:])
        if run.returncode==0:record["cubin_sha256"]=hashlib.sha256(out.read_bytes()).hexdigest()
        result["compiles"].append(record)
        (here/"sm90_compile.json").write_text(json.dumps(result,indent=2))
        print(threads,run.returncode,run.stdout[-600:],flush=True)
        if run.returncode:break
result["status"]="success" if len(result["compiles"])==2 and all(r["exit"]==0 for r in result["compiles"]) else "failed"
(here/"sm90_compile.json").write_text(json.dumps(result,indent=2))
