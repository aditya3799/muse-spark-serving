"""Launch the local adaptive verify trial in the pinned CUDA environment."""
import os
from pathlib import Path
import sys
here=Path(__file__).resolve().parent
cuda=Path(sys.prefix)/"lib/python3.12/site-packages/nvidia/cu13"
headers=Path(sys.prefix)/"python-headers/usr/include"
env=dict(os.environ,CUDA_HOME=str(cuda),CUDA_PATH=str(cuda),MAX_JOBS="1",HUMMING_DISABLE_PARALLEL_BUILD="1")
for name,dirs in (("PATH",[Path(sys.prefix)/"bin",cuda/"bin"]),("CPATH",[headers/"python3.12",headers]),("LD_LIBRARY_PATH",[cuda/"lib","/usr/lib/wsl/lib"])):
    env[name]=":".join(str(p) for p in dirs)+":"+env.get(name,"")
script="oracle_suite.py" if "--oracle-suite" in sys.argv else "repair_widths.py"
os.execve(sys.executable,[sys.executable,"-B","-u",str(here/script),*[a for a in sys.argv[1:] if a!="--oracle-suite"]],env)