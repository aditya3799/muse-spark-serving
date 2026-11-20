import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
import importlib.metadata as metadata
compiler=metadata.distribution('nvidia-cuda-nvcc');nvcc=next(Path(compiler.locate_file(f)).resolve() for f in compiler.files if str(f).endswith('/bin/nvcc'));cuda=nvcc.parent.parent
os.environ.update(CUDA_HOME=str(cuda),MAX_JOBS='1',SGLANG_JIT_LOG_RESOURCE_USAGE='1',HUMMING_DISABLE_PARALLEL_BUILD='1')
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+str(cuda/'bin')+os.pathsep+os.environ.get('PATH','')
os.environ['LD_LIBRARY_PATH']=str(cuda/'lib64')+':/usr/lib/wsl/lib:'+os.environ.get('LD_LIBRARY_PATH','')
headers=Path(sys.prefix)/'python-headers/usr/include';os.environ['CPATH']=str(headers/'python3.12')+':'+str(headers)+':'+os.environ.get('CPATH','')

out=P/'packed_bm128_01';out.mkdir(exist_ok=True);snap=out/'sources';snap.mkdir(exist_ok=True)
files=[P/'offline_packed_bf16_sm90.py',P/'probe_staged_adversarial.py',P/'adversarial_fixture.py',P/'staged_ops.py',P/'launch_bm128.py',P/'PACKED_EXPONENT_BOUNDARY.json',P/'ADVERSARIAL_CPU_CONTRACT.json']
files+=list((ROOT/'swarm_2026-11-16/moe/humming_vendor/humming').rglob('*.py'))+list((ROOT/'swarm_2026-11-16/moe/humming_vendor/humming/include').rglob('*.cuh'))
r={'status':'running','start':time.time(),'phases':[],'sources':{}}
for i,path in enumerate(files):
 data=path.read_bytes();name=str(i)+'_'+path.name;(snap/name).write_bytes(data);r['sources'][str(path)]={'snapshot':name,'sha256':hashlib.sha256(data).hexdigest()}
def save():(out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
def phase(label,script,extra,seconds):
 d=out/label;d.mkdir(exist_ok=True);cmd=['/usr/bin/timeout','--signal=TERM','--kill-after=5s',str(seconds)+'s',sys.executable,str(P/script),'--out',str(d),*extra]
 row={'label':label,'command':cmd,'start':time.time()};r['phases'].append(row);save()
 with (d/'stdout.log').open('w') as so,(d/'stderr.log').open('w') as se:
  proc=subprocess.Popen(cmd,stdout=so,stderr=se,env=dict(os.environ,PYTHONUNBUFFERED='1'));row['pid']=proc.pid;save();row['exit_code']=proc.wait()
 row['finish']=time.time();save();return row['exit_code']
save()
with (out/'telemetry.csv').open('w') as te:
 mon=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used,utilization.gpu','--format=csv','-l','1'],stdout=te,stderr=subprocess.DEVNULL)
 try:
  rc=phase('offline','offline_packed_bf16_sm90.py',['--compile','--reduced-m128','--count','4'],110)
  r['runtime_status']='NOT_REQUESTED_COMPILER_ONLY_LEASE'
  r.update(status='exited',finish=time.time());save()
 finally:
  mon.terminate()
  try:mon.wait(timeout=3)
  except subprocess.TimeoutExpired:mon.kill();mon.wait(timeout=3)
print(json.dumps({k:v for k,v in r.items() if k!='sources'}))
