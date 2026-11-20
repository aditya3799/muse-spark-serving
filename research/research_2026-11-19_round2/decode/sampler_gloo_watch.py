"""Bounded process-group memory monitoring for local CPU Gloo proof."""
import datetime,json,os,resource,signal,subprocess,sys,threading,time
from pathlib import Path
here=Path(__file__).resolve().parent;resource.setrlimit(resource.RLIMIT_CORE,(0,0))
def available():
 return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
before=available();print('preflight_available_bytes',before,flush=True)
assert before>=3*1024**3,'Insufficient free host RAM; leave workload unstarted'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
records=[];stop=threading.Event();aborted=[]
with (here/'sampler_gloo_stdout.log').open('w') as log:
 process=subprocess.Popen([sys.executable,'-u',str(here/'sampler_gloo_probe.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True,env=env)
 def monitor():
  while not stop.wait(1):
   free=available();rss=pss=0;count=0
   for path in Path('/proc').iterdir():
    if not path.name.isdigit():continue
    try:
     if os.getpgid(int(path.name))!=process.pid:continue
     count+=1;data=(path/'smaps_rollup').read_text().splitlines()
     rss+=int(next(x.split()[1] for x in data if x.startswith('Rss:')))*1024
     pss+=int(next(x.split()[1] for x in data if x.startswith('Pss:')))*1024
    except (ProcessLookupError,PermissionError,FileNotFoundError,StopIteration):pass
   records.append(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),available_bytes=free,group_rss_bytes=rss,group_pss_bytes=pss,processes=count))
   if free<768*1024**2:
    aborted.append('available host RAM below768MiB');os.killpg(process.pid,signal.SIGTERM);return
 thread=threading.Thread(target=monitor,daemon=True);thread.start()
 for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
 code=process.wait();stop.set();thread.join()
(here/'sampler_gloo_memory.json').write_text(json.dumps(dict(before_available_bytes=before,records=records,exit_code=code,aborted=aborted,gpu_used=False),indent=2))
sys.exit(code)
