"""Build the verified vendor sweep with FP32 destinations for explicit torch.mm(out_dtype) calls."""
from pathlib import Path
import hashlib
import subprocess
import sysconfig
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
def pinned(path, digest):
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest, str(path)
    return data.decode("utf-8")
probe = pinned(ROOT/"research_2026-11-18_rl/lt_probe.cc","caaeaf026ec51bd5eb94c356ed770931a1a2f68bfbdfc67e24e3aa1270440fe1")
sweep = pinned(ROOT/"research_2026-11-18_lt_audit/sweep.cc","81bbe151e46d533bfd1d5973264e2f05220247a1e550c32cac68aaa2b7ebe321")
for dest in ("output","parent_output"):
    use = "&p->" + dest + ", CUDA_R_16BF"
    assert probe.count(use)==1
    probe=probe.replace(use,use.replace("16BF","32F"))
assert sweep.count("&p->output,CUDA_R_16BF")==1
sweep=sweep.replace("&p->output,CUDA_R_16BF","&p->output,CUDA_R_32F")
assert sweep.count('#include "../research_2026-11-18_rl/lt_probe.cc"')==1
sweep=sweep.replace('#include "../research_2026-11-18_rl/lt_probe.cc"',probe)
toolkit=Path(sysconfig.get_paths()["purelib"])/"nvidia/cu13"
with tempfile.TemporaryDirectory() as folder:
    code=Path(folder)/"sweep_fp32.cc"
    code.write_text(sweep)
    subprocess.run(["g++","-O2","-std=c++17","-fPIC","-shared",str(code),
                    "-I"+str(toolkit/"include"),"-I"+str(toolkit/"include/cccl"),
                    "-L"+str(toolkit/"lib"),"-Wl,-rpath,"+str(toolkit/"lib"),"-l:libcublasLt.so.13",
                    "-o",str(HERE/"sweep_fp32.so")],check=True)

