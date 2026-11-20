"""Run local Humming scale-order trials with the frozen CUDA libraries."""
import os
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
cuda = Path(sys.prefix) / "lib/python3.12/site-packages/nvidia/cu13"
headers = Path(sys.prefix) / "python-headers/usr/include"
env = dict(os.environ, CUDA_HOME=str(cuda), CUDA_PATH=str(cuda), MAX_JOBS="1", HUMMING_DISABLE_PARALLEL_BUILD="1")
for key, paths in (("PATH", [Path(sys.prefix)/"bin", cuda/"bin"]), ("CPATH", [headers/"python3.12", headers]),
                   ("LD_LIBRARY_PATH", [cuda/"lib", "/usr/lib/wsl/lib"])):
    env[key] = ":".join(map(str,paths))+":"+env.get(key,"")
mode = sys.argv[1]
assert mode in ("offline", "paired", "paired_fused")
if mode == "offline":
    env["CUDA_VISIBLE_DEVICES"] = ""
os.execve(sys.executable,[sys.executable,"-u",str(here/(mode+".py")),*sys.argv[2:]],env)

