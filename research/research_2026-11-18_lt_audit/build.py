"""Build selected diagnostics against the installed PyTorch environment's CUDA libraries."""
from pathlib import Path
import argparse
import sysconfig
import subprocess
root = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("name", choices=("sweep","runtime_lt"))
name=p.parse_args().name
nvidia = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
options = [nvidia / "cu13", nvidia / "cublas"]
location=next((d for d in options if (d / "include/cublasLt.h").is_file()),None)
if location is None: raise RuntimeError("cuBLASLt headers not in the active environment")
libraries=list((location/"lib").glob("libcublasLt.so.*"))
if not libraries: raise RuntimeError("cuBLASLt shared library missing")
includes=[d/"include" for d in (location,nvidia/"cuda_runtime") if (d/"include").is_dir()]
cmd=["g++","-O2","-std=c++17","-shared","-fPIC"]
for inc in includes: cmd.extend(["-I",str(inc)])
cmd.extend([str(root / f"{name}.cc"),str(libraries[0]),"-Wl,-rpath,"+str(location/"lib"),
            "-o",str(root/f"{name}.so")])
subprocess.run(cmd,check=True)

