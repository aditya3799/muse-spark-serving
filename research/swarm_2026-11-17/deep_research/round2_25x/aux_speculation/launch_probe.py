import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
p=argparse.ArgumentParser();p.add_argument('--name',default='append_graph_01');p.add_argument('--seconds',type=int,default=170);a=p.parse_args()
out=P/a.name;out.mkdir(exist_ok=True);snap=out/'sources';snap.mkdir(exist_ok=True)
files=[P/'probe_append_graph.py',P/'launch_probe.py',ROOT/'sglang/python/sglang/srt/models/dflash.py',ROOT/'sglang/python/sglang/srt/mem_cache/memory_pool.py',ROOT/'sglang/python/sglang/kernels/ops/kvcache/cache_move.py',ROOT/'sglang/python/sglang/kernels/ops/speculative/fused_kv_materialize.py']
sources={}
for i,path in enumerate(files):
 data=path.read_bytes();name=str(i)+'_'+path.name;(snap/name).write_bytes(data);sources[str(path)]={'snapshot':name,'sha256':hashlib.sha256(data).hexdigest()}
cmd=['/usr/bin/timeout','--signal=TERM','--kill-after=5s',str(a.seconds)+'s',sys.executable,str(P/'probe_append_graph.py'),'--out',str(out/'results.json')]
r={'status':'running','start':time.time(),'command':cmd,'sources':sources}
def save():(out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
save()
with (out/'telemetry.csv').open('w') as te:
 mon=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used,utilization.gpu','--format=csv','-l','1'],stdout=te,stderr=subprocess.DEVNULL)
 try:
  with (out/'stdout.log').open('w') as so,(out/'stderr.log').open('w') as se:
   proc=subprocess.Popen(cmd,stdout=so,stderr=se,env=dict(os.environ,PYTHONUNBUFFERED='1'));r['pid']=proc.pid;save();rc=proc.wait()
  r.update(status='exited',finish=time.time(),exit_code=rc);save()
 finally:mon.terminate();mon.wait(timeout=5)
print(json.dumps({k:v for k,v in r.items() if k!='sources'}));raise SystemExit(rc)
