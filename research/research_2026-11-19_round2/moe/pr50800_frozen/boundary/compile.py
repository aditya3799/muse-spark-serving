"""Offline only: bounded complete WGS kernel compile, three coefficient variants."""
import hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
here=Path(__file__).resolve().parent
cuda=Path(sys.prefix)/'lib/python3.12/site-packages/nvidia/cu13'
source=here/'device_only.cu'
free={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
reads=[];stable=0;deadline=time.monotonic()+60
while time.monotonic()<deadline:
 free={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
 reads.append(dict(monotonic=time.monotonic(),available=free['MemAvailable']))
 stable=stable+1 if free['MemAvailable']>=int(2.3*1024**3) else 0
 if stable>=3:break
 time.sleep(1)
(here/'stability_preflight.json').write_text(json.dumps(dict(reads=reads,stable=stable,required_available=int(2.3*1024**3),bounded_seconds=60),indent=2))
assert stable>=3,('host memory guard',reads[-1])
report=dict(boundary_variant=True, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),device_execution=False,mem_available_start=free['MemAvailable'],address_space_limit=2*1024**3,variants=[])
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TMPDIR=str(here))
wrapper="import resource,os,sys; resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3)); resource.setrlimit(resource.RLIMIT_CORE,(0,0)); open('/proc/self/coredump_filter','w').write('0'); os.execv(sys.argv[1],sys.argv[1:])"
for variant in (0,):
 output=here/'boundary_sm90a.cubin'
 cmd=[str(cuda/'bin/nvcc'),'-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v', '-I'+str(cuda/'include'),'-I'+str(cuda/'include/cccl'),str(source),'-o',str(output)]
 start=time.time();p=subprocess.Popen([sys.executable,'-c',wrapper,'/usr/bin/time','-v',*cmd],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,start_new_session=True)
 timed_out=False
 try:log=p.communicate(timeout=120)[0]
 except subprocess.TimeoutExpired:
  timed_out=True;os.killpg(p.pid,signal.SIGKILL);log=p.communicate()[0]+'\nTIMEOUT120s\n'
 (here/'compile.log').write_text(log)
 row=dict(variant=variant,exit_code=p.returncode,seconds=time.time()-start,timed_out=timed_out,command=cmd)
 if p.returncode==0:
  row['cubin_sha256']=hashlib.sha256(output.read_bytes()).hexdigest();row['cubin_bytes']=output.stat().st_size
  dis=subprocess.run([str(Path(sys.prefix)/'lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump'),'--dump-sass',str(output)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=20,env=env)
  (here/'sass.txt').write_text(dis.stdout);row['disassembly_status']=dis.returncode
 report['variants'].append(row);(here/'compile.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
 if p.returncode:break
report['complete']=len(report['variants'])==1 and all(x['exit_code']==0 for x in report['variants'])
(here/'compile.json').write_text(json.dumps(report,indent=2))

