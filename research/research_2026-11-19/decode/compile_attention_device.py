"""Bounded CPU-only SM90a compilation, never launches a GPU kernel."""
import argparse,hashlib,json,os,resource,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('mode',choices=['request','flat']);a=p.parse_args()
here=Path(__file__).resolve().parent
cuda=Path('/home/aditya/.venvs/okc-marlin-tiles/lib/python3.12/site-packages/nvidia/cu13')
src=here/f'precision_attention_{a.mode}.cu';out=here/f'precision_attention_{a.mode}.o'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PATH=str(cuda/'bin')+':/usr/bin:/bin')
cmd=[str(cuda/'bin/nvcc'),'-std=c++17','-O3','-gencode=arch=compute_90a,code=sm_90a',
 '--expt-relaxed-constexpr','-Xptxas=-v','-c',str(src),'-o',str(out)]
def limits():
 resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
 resource.setrlimit(resource.RLIMIT_CPU,(60,65))
start=time.monotonic()
try:
 result=subprocess.run(cmd,capture_output=True,text=True,timeout=75,env=env,preexec_fn=limits)
 status=result.returncode;log=result.stdout+result.stderr
except subprocess.TimeoutExpired as exc:
 status='timeout';log=str(exc)
doc=dict(mode=a.mode,command=cmd,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
 returncode=status,wall_seconds=time.monotonic()-start,scope=__doc__,gpu_execution=False,
 memory_cap_bytes=3*1024**3,log=log,
 object_sha256=hashlib.sha256(out.read_bytes()).hexdigest() if out.exists() else None)
(here/f'precision_attention_{a.mode}_compile.json').write_text(json.dumps(doc,indent=2))
print(json.dumps(doc,indent=2))
