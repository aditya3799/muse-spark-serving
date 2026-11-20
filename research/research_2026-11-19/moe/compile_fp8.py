"""Compile an SM90a device kernel (no GPU access) using the local CUTLASS 4.2.1 snapshot."""
from pathlib import Path
import hashlib
import importlib.metadata as md
import json
import os
import re
import subprocess
import sys
import tempfile
import time
os.environ["CUDA_VISIBLE_DEVICES"]=""
d=Path(__file__).resolve().parent
root=d.parents[1]
include=root/"swarm_2026-11-17/deep_research/round3_structural/coordination/deepgemm_pinned/source/deep_gemm/include"
header="cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp"
latest=(d/"reference"/Path(header).name).read_bytes()
local=(include/header).read_bytes()
dist=md.distribution("nvidia-cuda-nvcc")
nvcc=next(Path(dist.locate_file(f)).resolve() for f in dist.files if str(f).endswith("/bin/nvcc"))
path=next(p for p in Path(sys.prefix).rglob("cuda_runtime.h") if "target" in str(p) or "nvidia" in str(p))
incs=[path.parent,path.parent/"cccl",nvcc.parent.parent/"include"]
incs.extend(p.parent.parent for p in Path(sys.prefix).rglob("cuda/std") if p.is_dir())
a={"toolchain":subprocess.check_output([str(nvcc),"--version"]).decode().splitlines()[-1],
   "snapshot":"CUTLASS 4.2.1, local DeepGEMM snapshot",
   "grouped_header_matches_upstream_bytes":latest==local,
   "upstream_header_sha256":hashlib.sha256(latest).hexdigest(),"local_header_sha256":hashlib.sha256(local).hexdigest(),
   "results":[]}
def compile_one(k,extra_dirs=(),defines=(),name="ordinary"):
    with tempfile.TemporaryDirectory() as tmp:
        target=Path(tmp)/"kernel.cubin"
        cmd=[str(nvcc),"-arch=sm_90a","--cubin","-std=c++17","--expt-relaxed-constexpr",
             "--expt-extended-lambda","-O2","-Xptxas=-v",f"-DPROBE_CTA_K={k}",
             *defines,*(f"-I{x}" for x in extra_dirs),f"-I{include}",*(f"-I{x}" for x in incs),
             str(d/"fp8_probe.cu"),"-o",str(target)]
        st=time.monotonic()
        rr=subprocess.run(cmd,capture_output=True,text=True)
        msg=rr.stdout+rr.stderr
        lines=[s[:380] for s in msg.splitlines() if re.search(r"error:|static assertion failed|fatal error|warning #|ptxas info.*Used|ptxas info.*spill",s)]
        rec=dict(mode=name,k=k,rc=rr.returncode,sec=time.monotonic()-st,diagnostics=lines[:12],
                 binary_size=target.stat().st_size if target.exists() else 0,
                 cubin_sha256=hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None)
        a["results"].append(rec)
        (d/"fp8_compile.json").write_text(json.dumps(a,indent=2))
        print(rec,flush=True)

if __name__=="__main__":
    for k in (16,64):
        compile_one(k)
