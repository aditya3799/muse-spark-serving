"""Compile the optional local cuBLASLt arithmetic probe with the active CUDA wheel."""
import os
import subprocess
import sys
from pathlib import Path

here=Path(__file__).resolve().parent
cuda=Path(sys.prefix)/"lib/python3.12/site-packages/nvidia/cu13"
inc,lib=cuda/"include",cuda/"lib"
assert (inc/"cublasLt.h").is_file(), inc
print(inc, flush=True)
subprocess.run(["/usr/bin/g++","-std=c++17","-O2","-shared","-fPIC","-I",str(inc),
    str(here/"lt_probe.cc"),"-L",str(lib),"-Wl,-rpath,"+str(lib),"-l:libcublasLt.so.13",
    "-o",str(here/"lt_probe.so")],check=True)
