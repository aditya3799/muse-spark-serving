"""Build the isolated CUDA consumer and inspect both target architectures offline."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
H=Path(__file__).resolve().parent
nvcc=str(Path(os.environ["CUDA_HOME"])/"bin/nvcc")
src=str(H/"gated_o.cu")
rows={}
with tempfile.TemporaryDirectory() as t:
 for arch in ("sm_86","sm_90a"):
    out=Path(t)/("kernel_"+arch+".cubin")
    ptx=out.with_suffix(".ptx")
    flags=[nvcc,"-std=c++17","-O3","-arch="+arch,src]
    subprocess.run(flags+["-cubin","-o",str(out)],check=True)
    subprocess.run(flags+["-ptx","-o",str(ptx)],check=True)
    p=ptx.read_text()
    mma=sorted(set(re.findall(r"(?:wmma|mma)\.mma[^; \n]+|mma\.sync[^; \n]+",p)))
    assert len(mma)>0 and any(".bf16" in i and ".f32" in i for i in mma),mma
    assert "shfl.sync.bfly" in p and "ex2.approx" in p and "rsqrt.approx" in p
    rows[arch]=dict(cubin_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),bytes=out.stat().st_size,
                    mma=mma,gate_shuffle=True,fast_exp_and_rsqrt=True)
so=H/"gated_o.so"
subprocess.run([nvcc,"-std=c++17","-O3","-shared","-Xcompiler=-fPIC","-gencode=arch=compute_86,code=sm_86",
               "-gencode=arch=compute_90a,code=sm_90a","--cudart=shared",src,"-o",str(so)],check=True)
(H/"offline.json").write_text(json.dumps(dict(sha256=hashlib.sha256(Path(src).read_bytes()).hexdigest(),architectures=rows,
                                  shared_object_sha256=hashlib.sha256(so.read_bytes()).hexdigest()),indent=2));print(rows)

