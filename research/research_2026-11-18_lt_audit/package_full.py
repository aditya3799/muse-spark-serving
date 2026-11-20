"""Build a self-contained source-only package for the empirically qualified full-width layer."""
from pathlib import Path
import ast
import hashlib
import json
import zipfile

here=Path(__file__).resolve().parent
dest=here/"full_projector"
dest.mkdir(exist_ok=True)
def save(name,text):
    (dest/name).write_text(text,encoding="utf-8")
base=(here.parent/"research_2026-11-18_rl/lt_probe.cc").read_text(encoding="utf-8-sig")
# Remove debug-only exported probes after the common descriptor factory.
base=base[:base.index('extern "C" int probe_count')]
cc=(here/"sweep.cc").read_text(encoding="utf-8-sig").replace('#include "../research_2026-11-18_rl/lt_probe.cc"',base)
save("vendor.cc",cc)
backend=(here/"native_abi.py").read_text(encoding="utf-8-sig")
ws=(here/"runtime_projection.py").read_text(encoding="utf-8-sig")
ws=ws[ws.index("@lru_cache(maxsize=2)"):ws.index("def compatible(")]
backend=backend.replace("import ctypes as ct","import ctypes as ct\nfrom functools import lru_cache\nimport torch")
save("backend.py",backend+"\n"+ws)
runtime=(here/"full_runtime.py").read_text(encoding="utf-8-sig")
runtime=runtime.replace("from native_abi import load_lib, describe\nfrom runtime_projection import native_workspace",
                        "from .backend import load_lib, describe, native_workspace")
save("__init__.py",runtime)
save("build.py",'''"""Build the package's optional native library locally from the active CUDA environment."""
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
''')
save("README.md",'''# Full-width projection tuner

This optional inference component wraps an existing unquantized, bias-free PyTorch Linear. Runtime cuBLASLt heuristics supply the reduction and eligible algorithm choices. The source contains no model dimensions, rank assumptions or frozen algorithm IDs. Compile the source once in the deployment environment with its actual CUDA library.

~~~bash
python full_projector/build.py
~~~

After checkpoint loading, create one wrapper for the existing layer, call prepare with distinct representative warmup and qualification tensors on its inference stream, and keep the existing hidden normalization after the wrapper. Unregistered shapes and unsupported paths execute the existing layer.

~~~python
from full_projector import CompleteProjection

projection = CompleteProjection(draft_model.fc.eval(), empirical=False)
with torch.inference_mode():
    qualification = projection.prepare(tuning_inputs, separate_validation_inputs)
    normalized = draft_model.hidden_norm(projection(target_hidden))
~~~

Default strict mode checks all output bits on every call and preserves the original result on divergence. Empirical mode skips the second GEMM, so it relies on the finite device-specific qualification. Inputs supplied to prepare must be representative of the loaded checkpoint and should include difficult cancellation cases. Qualification creates its own vendor handle and workspace; startup wall time, paired native/optimized times, and the number of calls to recover the setup cost are returned. Unsupported cuBLAS/PyTorch builds naturally remain on the native layer when its vendor workspace cannot be resolved. Call close after destroying dependent CUDA graphs. Invalidate and prepare again after any in-place weight reload, especially for inference tensors without version counters.

This source-only prototype was developed from local SM86 single-GPU tests. Native SM90 H200 comparison, full-model greedy output validation, deployment cold start and peak memory are outstanding. No hardware-specific configuration should be copied to a different device or CUDA installation.
''')
files=["__init__.py","backend.py","build.py","vendor.cc","README.md"]
for name in files:
    if name.endswith(".py"): ast.parse((dest/name).read_text(encoding="utf-8-sig"))
with zipfile.ZipFile(here/"full_projector_source.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
    for name in files: archive.write(dest/name,"full_projector/"+name)
print("source package",hashlib.sha256((here/"full_projector_source.zip").read_bytes()).hexdigest())

