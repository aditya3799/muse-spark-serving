import hashlib,json,os,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent
out=P/'gpu_integration_01';out.mkdir(exist_ok=True)
files=list(P.glob('*.py'))+list((P/'candidate_sources').rglob('*.py'))
snap=out/'source_snapshots';snap.mkdir(exist_ok=True)
sources={}
for i,p in enumerate(files):
 name='%02d_%s'%(i,p.name);b=p.read_bytes();(snap/name).write_bytes(b);sources[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'snapshot':name}
receipt={'status':'running','start':time.time(),'sources':sources,'runs':[]}
def save(): (out/'run_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
save()
telemetry=(out/'telemetry.csv').open('w')
monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used,utilization.gpu','--format=csv','-l','1'],stdout=telemetry,stderr=subprocess.DEVNULL)
code=0
try:
 jobs=[('integrated_methods',[str(P/'test_integrated_methods.py'),'--device','cuda','--out',str(out/'integrated_methods.json')])]
 jobs += [(m,[str(P/'gpu_integration_probe.py'),'--mode',m,'--out',str(out/(m+'.json'))]) for m in ('race','baseline','candidate')]
 for name,args in jobs:
  command=['/usr/bin/timeout','--signal=TERM','--kill-after=5s','65s',sys.executable,*args]
  run={'name':name,'command':command,'start':time.time()};receipt['runs'].append(run);save()
  with (out/(name+'_stdout.log')).open('w') as stdout,(out/(name+'_stderr.log')).open('w') as stderr:
   process=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=dict(os.environ,PYTHONUNBUFFERED='1',MAX_JOBS='1'));run['pid']=process.pid;save();rc=process.wait()
  run.update(exit_code=rc,finish=time.time());save();print(json.dumps(run),flush=True)
  if rc:code=rc;break
finally:
 monitor.terminate();monitor.wait(timeout=5);telemetry.close();receipt.update(status='exited',exit_code=code,finish=time.time());save()
raise SystemExit(code)
