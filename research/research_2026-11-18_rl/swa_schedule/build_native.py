"""Compile the pinned CUDA scheduler kernel intact for local execution and SM90a code generation."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
here=Path(__file__).resolve().parent
raw=(here/"flash_prepare_scheduler.ref.cu").read_text()
kernel=raw[raw.index("namespace flash {"):raw.index("} // flash")+len("} // flash")]
includes=raw[:raw.index('#include "flash.h"')]
host=r"""
extern "C" int okc_plan(const int* counts,const int* cumulative,int* out,int b,int q,
                        int cap,int bm,int bn,int sms,int splits,cudaStream_t stream){
  int rounded=(b+3)/4*4;
  if(b<1 || b>62 || splits<1 || !counts || !cumulative || !out) return -1;
  // The local-window path uses unsorted variable-length batches and packed GQA.
  #define OKC_CALL(W) flash::prepare_varlen_num_blocks_kernel<W,false><<<1,W*32,0,stream>>>( \
      q,cap,0,cumulative,nullptr,nullptr,nullptr,counts,nullptr,b,2,8,sms,splits, \
      cutlass::FastDivmod(bm),cutlass::FastDivmod(bn),out+3*rounded,out+rounded, \
      out,nullptr,out+2*rounded,false,false,true,8*1024*1024/(bn*128))
  if(b<=31){OKC_CALL(1);}else{OKC_CALL(2);}
  #undef OKC_CALL
  return cudaGetLastError();
}

__global__ void observe_plan(const int* p,int* audit,int b,int rounded){
  int i=threadIdx.x;
  if(i<b){
    audit[i]=p[i];
    audit[b+i]=p[rounded+i];
    audit[2*b+i]=p[2*rounded+i];
  }
}
extern "C" int okc_observe(const int* p,int* receipt,int b,cudaStream_t stream){
  observe_plan<<<1,64,0,stream>>>(p,receipt,b,(b+3)/4*4);
  return cudaGetLastError();
}
"""
cubin=here/"scheduler_native.so"
nvcc=next(Path(importlib.metadata.distribution("nvidia-cuda-nvcc").locate_file(x)).resolve() for x in importlib.metadata.distribution("nvidia-cuda-nvcc").files if str(x).endswith("/bin/nvcc"))
headers=next(p for p in Path(sys.prefix).rglob("cuda_runtime.h") if "nvidia" in str(p))
inc=[here.parents[1]/"swarm_2026-11-17/deep_research/round3_structural/coordination/deepgemm_pinned/source/deep_gemm/include",headers.parent,headers.parent/"cccl",nvcc.parent.parent/"include"]
inc.extend(p.parent.parent for p in Path(sys.prefix).rglob("cuda/std") if p.is_dir())
rt=next(p for p in Path(sys.prefix).rglob("libcudart.so.*") if p.is_file())
with tempfile.TemporaryDirectory() as folder:
    src=Path(folder)/"scheduler.cu";src.write_text(includes+kernel+"\n"+host)
    cmd=[str(nvcc),"-std=c++17","-O2","-shared","-Xcompiler=-fPIC","--expt-relaxed-constexpr","--extended-lambda",
         "-gencode=arch=compute_86,code=sm_86","-gencode=arch=compute_90a,code=sm_90a",
         *(f"-I{p}" for p in inc),"-cudart=none",str(src),f"-L{rt.parent}",f"-l:{rt.name}",f"-Xlinker=-rpath={rt.parent}","-o",str(cubin)]
    res=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (here/"compile.log").write_text(res.stdout)
    if res.returncode:
        print(res.stdout[-2300:]);sys.exit(res.returncode)
out=dict(upstream="sgl-project/sgl-flash-attn@f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/flash_prepare_scheduler.cu",
    kernel_sha256=hashlib.sha256(kernel.encode()).hexdigest(),library_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(),
    library_bytes=cubin.stat().st_size,targets=["sm_86","sm_90a"],binary="Local benchmark only")
(here/"compile.json").write_text(json.dumps(out,indent=2));print(out)
