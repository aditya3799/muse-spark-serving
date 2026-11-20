"""Bounded offline compiler for the small coefficient oracle; no Torch import."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
here=Path(__file__).resolve().parent
cuda=Path(sys.prefix)/'lib/python3.12/site-packages/nvidia/cu13'
source=here/'register_dequant_probe.cu';output=here/'libregister_dequant_sm86.so'
cmd=[str(cuda/'bin/nvcc'),'-std=c++17','-O3','-arch=sm_86','--shared','-Xcompiler=-fPIC','-Xptxas=-v',
     '-I'+str(cuda/'include'),'-I'+str(cuda/'include/cccl'),str(source),'-o',str(output),
     '-L'+str(cuda/'lib'),'-l:libcudart.so.13','-Xlinker=-rpath','-Xlinker='+str(cuda/'lib')]
start=time.time();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TMPDIR=str(here))
process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,start_new_session=True)
try:text=process.communicate(timeout=60)[0]
except subprocess.TimeoutExpired:
 os.killpg(process.pid,signal.SIGKILL);text=process.communicate()[0]+'\nTIMEOUT60s\n'
(here/'register_dequant_compile.log').write_text(text)
report=dict(exit_code=process.returncode,elapsed_seconds=time.time()-start,command=cmd,
 source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),address_space_limit_bytes=2*1024**3)
if process.returncode==0:report['library_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
(here/'register_dequant_compile.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
sys.exit(process.returncode)
