"""Build a linked SM90a C ABI library, without using GPU hardware."""
from pathlib import Path
import json
import hashlib
import os
import subprocess
import sys
import tempfile
import compile_fp8 as cfg

d=Path(__file__).resolve().parent
include=d/"cutlass_pinned/include"
shared="--shared-weights" in sys.argv
label="host_shared" if shared else "host_build"
out=d/("libgroup_shared_sm90a.so" if shared else "libgroup_g16_sm90a.so")
with tempfile.TemporaryDirectory() as folder:
    tmp=Path(folder)
    rel=Path("include/cutlass/detail/collective/mixed_input_utils.hpp")
    (tmp/rel).parent.mkdir(parents=True)
    (tmp/rel).write_bytes((include/Path(*rel.parts[1:])).read_bytes())
    subprocess.run(["patch","-p1","-i",str(d/"custom_fragment.diff")],cwd=tmp,check=True)
    cudart=next(p for p in Path(sys.prefix).rglob("libcudart.so.*") if p.is_file())
    cli=[str(cfg.nvcc),"-std=c++17","-arch=sm_90a","--shared","-Xcompiler=-fPIC",
         "--expt-extended-lambda","--expt-relaxed-constexpr","-Xptxas=-v","-O2",
         "-DPACKED_FP4","-DPACKED_SCALE","-DPROBE_CTA_K=64",f"-I{tmp/'include'}",
         f"-I{include}",*(f"-I{x}" for x in cfg.incs),"-cudart=none",str(d/"group_host.cu"),
         f"-L{cudart.parent}",f"-l:{cudart.name}",f"-Xlinker=-rpath={cudart.parent}",
         "-L/usr/lib/wsl/lib","-lcuda","-o",str(tmp/"lib.so")]
    if shared:cli.insert(1,"-DSHARED_HUMMING_WEIGHT")
    result=subprocess.run(cli,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (d/(label+".log")).write_text(result.stdout)
    if result.returncode:
        print("\n".join(line[:460] for line in result.stdout.splitlines() if "error" in line.lower())[:1600])
        sys.exit(result.returncode)
    data=(tmp/"lib.so").read_bytes()
    out.write_bytes(data)
    info=dict(target="sm_90a",CUTLASS="cd9d8a7d612b6b9ad0f1b698925e31243e828629",
              bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
              symbols=subprocess.check_output(["nm","-D","--defined-only",str(out)],text=True).splitlines())
    info['symbols']=[s.split()[-1] for s in info['symbols'] if ' okc_' in s]
    (d/(label+".json")).write_text(json.dumps(info,indent=2))
    print(info)
