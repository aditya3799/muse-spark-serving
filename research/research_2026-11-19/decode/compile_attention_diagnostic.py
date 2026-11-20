"""Bounded CPU-only Hopper diagnostic CABI build and layout check; no launches."""
import ctypes,hashlib,json,os,resource,subprocess,time
from pathlib import Path
here=Path(__file__).resolve().parent
cuda=Path('/home/aditya/.venvs/okc-marlin-tiles/lib/python3.12/site-packages/nvidia/cu13')
src=here/'attention_diagnostic_native.cu';out=here/'attention_diagnostic_native.so'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PATH=str(cuda/'bin')+':/usr/bin:/bin')
cmd=[str(cuda/'bin/nvcc'),'-std=c++17','-O3','-gencode=arch=compute_90a,code=sm_90a',
 '--expt-relaxed-constexpr','-Xptxas=-v','-shared','-Xcompiler=-fPIC',str(src),'-o',str(out)]
def limits():
 resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
 resource.setrlimit(resource.RLIMIT_CPU,(60,65))
start=time.monotonic()
try:
 result=subprocess.run(cmd,capture_output=True,text=True,timeout=75,env=env,preexec_fn=limits)
 status=result.returncode;log=result.stdout+result.stderr
except subprocess.TimeoutExpired as exc:status='timeout';log=str(exc)
doc=dict(scope=__doc__,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),command=cmd,
 returncode=status,wall_seconds=time.monotonic()-start,memory_cap_bytes=3*1024**3,cores_disabled=True,
 gpu_execution=False,serving_qualified=False,log=log)
if status==0:
 doc.update(library_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),library_bytes=out.stat().st_size)
 try:
  from hopper_attention_diagnostic import Params,load
  lib=load(out)
  doc.update(abi_layout_passed=True,params_size=ctypes.sizeof(Params),
   fields={name:getattr(Params,name).offset for name,_ in Params._fields_})
 except Exception as exc:
  doc.update(abi_layout_passed=False,abi_exception=repr(exc))
(here/'attention_diagnostic_compile.json').write_text(json.dumps(doc,indent=2))
print(json.dumps({k:v for k,v in doc.items() if k not in ('log','fields','command')},indent=2))
if status!=0:print(log[-6000:]);raise SystemExit(1)
if not doc.get('abi_layout_passed'):raise SystemExit(2)
