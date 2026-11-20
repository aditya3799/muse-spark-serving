"""Exclusive reservation required. Hard bounded Triton-only diagnostic."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent
pa=argparse.ArgumentParser();pa.add_argument('--out',type=Path,required=True);pa.add_argument('--seconds',type=int,default=180);a=pa.parse_args()
assert 30<=a.seconds<=180
a.out.mkdir(parents=True,exist_ok=True);snap=a.out/'source_snapshots';snap.mkdir(exist_ok=True)
files=list(P.glob('*.py'))+[P/'cpu_protocol_results.json',P.parents[2]/'sglang/python/sglang/kernels/ops/speculative/dflash.py']
pins={}
for i,f in enumerate(files):
    name='%02d_%s'%(i,f.name);(snap/name).write_bytes(f.read_bytes())
    pins[str(f)]={'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'snapshot':name}
cmd=['/usr/bin/timeout','--signal=TERM','--kill-after=10s',str(a.seconds)+'s',sys.executable,str(P/'prefix_gpu_probe.py'),'--out',str(a.out)]
r={'status':'started','started':time.time(),'command':cmd,'sources':pins}
def save():
    (a.out/'run_receipt.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
save()
with (a.out/'stdout.log').open('w') as out,(a.out/'stderr.log').open('w') as err:
    child=subprocess.Popen(cmd,stdout=out,stderr=err,env=dict(os.environ,PYTHONUNBUFFERED='1',MAX_JOBS='1'))
    r['pid']=child.pid;save();code=child.wait()
r.update(status='exited',exit_code=code,finished=time.time());save();print(json.dumps(r));raise SystemExit(code)
