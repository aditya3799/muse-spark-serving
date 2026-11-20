"""Durably supervise one bounded GPU probe; preserve original child exit."""
from pathlib import Path
import argparse,hashlib,importlib.metadata as metadata,json,os,subprocess,sys,time
P=Path(__file__).resolve().parent
pa=argparse.ArgumentParser();pa.add_argument('--script',default='fragment_probe.py');pa.add_argument('--seconds',type=int,default=180);pa.add_argument('--out',type=Path,required=True);a,rest=pa.parse_known_args()
if not 10<=a.seconds<=600:raise ValueError('Bound must be10..600seconds')
script=(P/a.script).resolve()
if script.parent!=P or script.suffix!='.py':raise ValueError('Only phase4 scripts allowed')
a.out.mkdir(parents=True,exist_ok=True)
compiler=metadata.distribution('nvidia-cuda-nvcc');nvcc=next(Path(compiler.locate_file(f)).resolve() for f in compiler.files if str(f).endswith('/bin/nvcc'));cuda=nvcc.parent.parent
os.environ.update(CUDA_HOME=str(cuda),MAX_JOBS='1',HUMMING_DISABLE_PARALLEL_BUILD='1',PYTHONUNBUFFERED='1',SGLANG_CACHE_DIR=str(Path.home()/'.cache/okc-marlin-tiles'),SGLANG_JIT_CACHE_DIR=str(Path.home()/'.cache/okc-marlin-tiles/jit'))
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+str(cuda/'bin')+os.pathsep+os.environ.get('PATH','')
os.environ['LD_LIBRARY_PATH']=str(cuda/'lib64')+':/usr/lib/wsl/lib:'+os.environ.get('LD_LIBRARY_PATH','')
headers=Path(sys.prefix)/'python-headers/usr/include';os.environ['CPATH']=str(headers/'python3.12')+':'+str(headers)+':'+os.environ.get('CPATH','')
command=['/usr/bin/timeout','--signal=TERM','--kill-after=15s',str(a.seconds)+'s',sys.executable,str(script),'--out',str(a.out)]+rest
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r={'status':'started','supervisor_pid':os.getpid(),'started_utc_seconds':time.time(),'command':command,'script_sha256':sha(script),'launcher_sha256':sha(Path(__file__)),'python':sys.executable,'nvcc':str(nvcc),'MAX_JOBS':'1','HUMMING_DISABLE_PARALLEL_BUILD':'1'}
def gpu_telemetry():
 command=[('/usr/lib/wsl/lib/nvidia-smi' if Path('/usr/lib/wsl/lib/nvidia-smi').exists() else 'nvidia-smi'),
  '--query-gpu=index,name,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,memory.used,utilization.gpu',
  '--format=csv,noheader,nounits']
 try:
  outcome=subprocess.run(command,capture_output=True,text=True,timeout=5)
  return {'utc_seconds':time.time(),'monotonic_seconds':time.monotonic(),'command':command,'exit_code':outcome.returncode,'stdout':outcome.stdout,'stderr':outcome.stderr}
 except Exception as error:
  return {'utc_seconds':time.time(),'error':repr(error),'command':command}
r['started_monotonic_seconds']=time.monotonic()
r['diagnostic_environment']={key:os.environ.get(key) for key in ('CUDA_LAUNCH_BLOCKING','CUDA_VISIBLE_DEVICES','TRITON_INTERPRET','TRITON_DEBUG','TORCH_USE_CUDA_DSA','CUDA_INJECTION64_PATH','NV_COMPUTE_PROFILER_INJECTION_PATH','LD_PRELOAD','CUDA_MODULE_LOADING','CUDA_DEVICE_MAX_CONNECTIONS','SGLANG_JIT_LOG_RESOURCE_USAGE')}
r['gpu_before']=gpu_telemetry()
snapshots=a.out/'source_snapshots';snapshots.mkdir(exist_ok=True)
r['source_snapshots']={}
for source_file in sorted(list(P.glob('*.py'))+list(P.glob('*.cuh'))):
 snapshot=snapshots/source_file.name;snapshot.write_bytes(source_file.read_bytes());r['source_snapshots'][source_file.name]=sha(snapshot)
def save():
 temp=a.out/'run_receipt.tmp';temp.write_text(json.dumps(r,indent=2)+'\n');temp.replace(a.out/'run_receipt.json')
save()
with (a.out/'stdout.log').open('w') as stdout,(a.out/'stderr.log').open('w') as stderr:
 child=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True);r['child_pid']=child.pid;save();print(json.dumps(r),flush=True)
 code=child.wait()
r.update(status='exited',exit_code=code,finished_utc_seconds=time.time(),finished_monotonic_seconds=time.monotonic())
r['gpu_after']=gpu_telemetry()
result=a.out/'results.json'
if result.exists():r['report_sha256']=sha(result)
save();print(json.dumps(r),flush=True);sys.exit(code)
