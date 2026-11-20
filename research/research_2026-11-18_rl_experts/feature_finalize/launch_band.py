"""Launch the local feature-sharded expert trial in the pinned CUDA environment."""
import os
from pathlib import Path
import sys
here=Path(__file__).resolve().parent
cuda=Path(sys.prefix)/"lib/python3.12/site-packages/nvidia/cu13"
headers=Path(sys.prefix)/"python-headers/usr/include"
env=dict(os.environ,CUDA_HOME=str(cuda),CUDA_PATH=str(cuda),MAX_JOBS="1",HUMMING_DISABLE_PARALLEL_BUILD="1")
for name,dirs in (("PATH",[Path(sys.prefix)/"bin",cuda/"bin"]),("CPATH",[headers/"python3.12",headers]),("LD_LIBRARY_PATH",[cuda/"lib","/usr/lib/wsl/lib"])):
    env[name]=":".join(str(p) for p in dirs)+":"+env.get(name,"")
os.execve(sys.executable,[sys.executable,"-B","-u",str(here/"warp_native.py"),*sys.argv[1:]],env)