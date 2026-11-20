"""Run a local offline compile or a paired GPU measurement with the pinned CUDA toolchain."""
import json
import os
import subprocess
import sys
from pathlib import Path

here=Path(__file__).resolve().parent
root=here.parent
env=os.environ.copy()
cuda=Path(sys.prefix)/"lib/python3.12/site-packages/nvidia/cu13"
headers=Path(sys.prefix)/"python-headers/usr/include"
env.update(CUDA_HOME=str(cuda),CUDA_PATH=str(cuda),MAX_JOBS="1",
           PATH=str(Path(sys.prefix)/"bin")+":"+str(cuda/"bin")+":"+env.get("PATH",""),
           CPATH=str(headers/"python3.12")+":"+str(headers)+":"+env.get("CPATH",""),
           LD_LIBRARY_PATH=str(cuda/"lib")+":/usr/lib/wsl/lib:"+env.get("LD_LIBRARY_PATH",""))
if sys.argv[1]=="offline":
    env["CUDA_VISIBLE_DEVICES"]=""
    old=root/"swarm_2026-11-16/moe/phase2/scale_activation_offline_sm90"
    unit=(old/"kernel.cu").read_text().replace(str(old.parent/"scale_activation.cuh"),
          str(here/"route_activation.cuh")).replace("ScaledActivationKernel<bf16_t, true>",
          "RouteActivationKernel<bf16_t, true, 8>")
    (here/"route_sm90.cu").write_text(unit)
    archived=json.loads((old/"report.json").read_text())["compiler"]["command"]
    cmd=[str(cuda/"bin/nvcc"),"--cubin","--gpu-architecture=sm_90a","-DSGL_CUDA_ARCH=900","-std=c++20",
         "-O3","--expt-relaxed-constexpr","--use_fast_math","-Xptxas=-v",
         *[v for v in archived if v.startswith("-I")],str(here/"route_sm90.cu"),"-o",str(here/"route_sm90.cubin")]
    with (here/"offline.log").open("w") as f:
        subprocess.run(cmd,env=env,stdout=f,stderr=f,check=True,timeout=100)
else:
    subprocess.run([sys.executable,"-u",str(here/"paired.py"),*sys.argv[1:]],env=env,check=True)


