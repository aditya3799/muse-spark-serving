"""Receipt-preserving <=180s native diagnostic launcher. No build commands."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pa=argparse.ArgumentParser();pa.add_argument('--out',type=Path,required=True);pa.add_argument('--seconds',type=int,default=180)
args=pa.parse_args();assert 30<=args.seconds<=180
args.out.mkdir(parents=True,exist_ok=True)
script=P/'native_probe.py';manifest=json.loads((P/'manifest.json').read_text())
files=[script,Path(__file__),P/'manifest.json',P/'NATIVE_CONTRACT_REVIEW.json',P/'test_lifecycle.py',P/'cpu_lifecycle_validation.json']
for rel,digest in manifest['candidate_sources'].items():
    source=P/'candidate_sources'/rel;assert sha(source)==digest;files.append(source)
files.extend((P/'native_sources').glob('*'))
snap=args.out/'source_snapshots';snap.mkdir(exist_ok=True)
pins={}
for index,source in enumerate(files):
    target=snap/('%02d_%s'%(index,source.name));target.write_bytes(source.read_bytes())
    pins[str(source)]={'sha256':sha(source),'snapshot':target.name}
command=['/usr/bin/timeout','--signal=TERM','--kill-after=10s',str(args.seconds)+'s',sys.executable,str(script),'--out',str(args.out)]
receipt={'status':'started','started':time.time(),'supervisor_pid':os.getpid(),'command':command,'sources':pins,'scope':'Exclusive GPU reservation required. Native prebuilt FA3 only; no extension compilation. Unsupported exit3, failed exit1, timeout124; none imply pass.'}
def save():
    temp=args.out/'run_receipt.tmp';temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(args.out/'run_receipt.json')
save();os.environ['PYTHONUNBUFFERED']='1';os.environ['MAX_JOBS']='1'
with (args.out/'stdout.log').open('w') as stdout,(args.out/'stderr.log').open('w') as stderr:
    child=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True)
    receipt['child_pid']=child.pid;save();print(json.dumps(receipt),flush=True);code=child.wait()
receipt.update(status='exited',exit_code=code,finished=time.time())
for name in ('results.json','stdout.log','stderr.log'):
    if (args.out/name).exists():receipt[name+'_sha256']=sha(args.out/name)
save();print(json.dumps(receipt),flush=True);raise SystemExit(code)
