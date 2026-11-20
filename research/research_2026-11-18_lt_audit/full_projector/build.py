"""Build the package's optional native library locally from the active CUDA environment."""
from pathlib import Path
import glob
import os
import shutil
import subprocess
import sysconfig

root=Path(__file__).resolve().parent
nvidia=Path(sysconfig.get_paths()["purelib"])/"nvidia"
prefixes=[nvidia/"cu13",nvidia/"cublas",Path(os.getenv("CUDA_HOME","/usr/local/cuda"))]
prefix=next((d for d in prefixes if (d/"include/cublasLt.h").is_file()),None)
if prefix is None: raise RuntimeError("installed cuBLASLt development headers unavailable")
locations=[prefix/"lib",prefix/"lib64",prefix/"targets/x86_64-linux/lib"]
libraries=[f for d in locations for f in sorted(d.glob("libcublasLt.so*"))]
if not libraries: raise RuntimeError("installed cuBLASLt shared library unavailable")
lib=libraries[0]
incdirs=[prefix/"include",nvidia/"cuda_runtime/include",prefix/"include/cccl"]
cxx=shutil.which(os.getenv("CXX","g++"))
if not cxx: raise RuntimeError("host compiler unavailable")
args=[cxx,"-O2","-std=c++17","-fPIC","-shared"]
for d in incdirs:
    if d.is_dir(): args+=["-I",str(d)]
args += [str(root/"vendor.cc"),str(lib),"-Wl,-rpath,"+str(lib.parent),"-o",str(root/"sweep.so")]
subprocess.run(args,check=True)
