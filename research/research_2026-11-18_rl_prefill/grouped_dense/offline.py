"""Compile the exact release-wheel Hopper grouped BF16 kernel without CUDA device access."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument("--headers",type=Path,required=True)
a=p.parse_args()
header=a.headers/"deep_gemm/impls/sm90_bf16_gemm.cuh"
wheel_sha="d1e3e1421d6624a16f32f2dee35c026d64d2f77799e5c0c5eaba68c27023630c"
results=[]
with tempfile.TemporaryDirectory() as t:
    root=Path(t)
    for n in [8192,4096]:
        source=root/f"consumer_{n}.cu"
        source.write_text(f"""#include <deep_gemm/impls/sm90_bf16_gemm.cuh>
using namespace deep_gemm;
void* grouped_symbol() {{
 return reinterpret_cast<void*>(&sm90_bf16_gemm_impl<
   cute::UMMA::Major::K,cute::UMMA::Major::K, 0,{n},4096, 8,
   128,128,64, 128,128,128, 4,128,256, 1,false,132,
   GemmType::MGroupedContiguousWithPsumLayout,false,cutlass::bfloat16_t>);
}}
""")
        opts=[str(Path(os.environ["CUDA_HOME"])/"bin/nvcc"),"-std=c++20","--expt-relaxed-constexpr","-O3","-arch=sm_90a","-I"+str(a.headers),str(source)]
        ptx=root/f"kernel_{n}.ptx"
        cubin=root/f"kernel_{n}.cubin"
        subprocess.run(opts+["-ptx","-o",str(ptx)],check=True)
        subprocess.run(opts+["-cubin","-o",str(cubin)],check=True)
        text=ptx.read_text()
        wgmma=sorted(set(re.findall(r"wgmma\.mma_async[^; \n]+",text)))
        assert any(".f32.bf16.bf16" in op for op in wgmma), wgmma
        assert ".entry" in text and ("cvt.rn.bf16" in text or "cvt.rn.bf16x2" in text)
        results.append(dict(n=n,k=4096,groups=8,sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(),
                            bytes=cubin.stat().st_size,wgmma=wgmma))
report=dict(wheel_sha256=wheel_sha,header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
            arch="sm90a",kernels=results,scope="Exact wheel's real CUDA kernel specialization; static reference config, native runtime heuristic and numerical outputs unmeasured.")
(HERE/"native_sm90.json").write_text(json.dumps(report,indent=2));print(report)

