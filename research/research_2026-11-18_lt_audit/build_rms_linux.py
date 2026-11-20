"""Compile the exact pinned FlashInfer CUDA RMSNorm with the local CUDA wheel."""
import hashlib
import os
import pathlib
import subprocess
import sysconfig
here=pathlib.Path(__file__).resolve().parent
source=here.parent/"research_2026-11-18_rl_dflash/rms_bridge.cu"
text=source.read_text()
assert text.count("__declspec(dllexport)")==1
(here/"rms_linux.cu").write_text(text.replace("__declspec(dllexport)",""))
cu=pathlib.Path(sysconfig.get_paths()["purelib"])/"nvidia/cu13"
inc=next(pathlib.Path("/mnt/c/Users").glob("*/AppData/Local/Temp/flashinfer-0.7.2/include/flashinfer/norm.cuh")).parents[1]
archive=inc.parents[1]/"okc_flashinfer_0_7_2_src.tar.gz"
assert hashlib.sha256(archive.read_bytes()).hexdigest()=="51a02aec7901e830fee69df4ed017d6bd30cbeac55902f7b2bce44b463539c18"
env=dict(os.environ)
env["PATH"]=str(cu/"bin")+":"+env["PATH"]
subprocess.run([str(cu/"bin/nvcc"),"--shared","-O3","-std=c++17","-arch=sm_86",
                "--expt-relaxed-constexpr","-Xcompiler=-fPIC","-I",str(inc),
                "-I",str(cu/"include/cccl"),"-L",str(cu/"lib"),"--cudart=shared",
                "-Xlinker","-rpath","-Xlinker",str(cu/"lib"),str(here/"rms_linux.cu"),
                "-o",str(here/"rms_linux.so")],check=True,env=env)

