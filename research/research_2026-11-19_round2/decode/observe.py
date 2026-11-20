"""Small local-only subprocess and GPU telemetry receipt wrapper."""
import datetime,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
here=Path(__file__).resolve().parent
label,script,*args=sys.argv[1:]
assert label.replace('_','').replace('-','').isalnum()
directory=here/('run_'+label);directory.mkdir(exist_ok=False)
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
cmd=[sys.executable,'-u',str(here/script),*args]
sources={str(p.relative_to(here)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((here/'prologue_package').glob('*.py'))+[here/script,here/'chain_entry.py']}
record=dict(command=cmd,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sources,remote=False)
(directory/'receipt.json').write_text(json.dumps(record,indent=2))
with (directory/'gpu_telemetry.csv').open('w') as telemetry,(directory/'stdout.log').open('w') as log:
 monitor=subprocess.Popen(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=timestamp,name,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used','--format=csv','-l','1'],stdout=telemetry,stderr=telemetry)
 try:
  process=subprocess.Popen(cmd,cwd=here.parents[1],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  for line in process.stdout:
   log.write(line);log.flush();print(line,end='',flush=True)
  code=process.wait()
 finally:
  monitor.terminate();monitor.wait()
record.update(exit_code=code,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(directory/'receipt.json').write_text(json.dumps(record,indent=2))
sys.exit(code)
