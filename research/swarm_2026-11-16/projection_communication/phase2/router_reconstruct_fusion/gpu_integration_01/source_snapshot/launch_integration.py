"""Bound one local component run and retain exact source snapshots/exit status."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.metadata as md,json,os,shutil,signal,subprocess,sys,time
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
base=Path(__file__).resolve().parent;helper=base/'probe_integration.py'
if a.out.exists():raise SystemExit('Fresh output directory required')
a.out.mkdir(parents=True);snapshot=a.out/'source_snapshot';snapshot.mkdir()
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
files=[base/'integration_manifest.json',Path(__file__),helper]+list((base/'integration_sources').rglob('*.py'))+list((base/'integration_sources').rglob('*.cuh'))
tracked={}
for source in files:
    dest=snapshot/source.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);tracked[str(source)]=sha(source)
manifest=json.loads((base/'integration_manifest.json').read_text())
for rel in manifest['sources']:
    frozen=a.workspace/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources'/rel
    source=frozen if frozen.exists() else a.workspace/'sglang'/rel;dest=snapshot/'reference_sources'/rel
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);tracked[str(source)]=sha(source)
for rel in ('optimization_marlin_tiles_2026-11-15/kernel_runtime.py','swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang/srt/models/muse_spark.py'):
    source=a.workspace/rel;dest=snapshot/'reference_sources'/rel
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);tracked[str(source)]=sha(source)
compiler=md.distribution('nvidia-cuda-nvcc');nvcc=next(Path(compiler.locate_file(f)).resolve() for f in compiler.files if str(f).endswith('/bin/nvcc'));cuda=nvcc.parent.parent
headers=Path(sys.prefix)/'python-headers/usr/include'
os.environ.update(RECONSTRUCT_BOUND='240',MAX_JOBS='1',HUMMING_DISABLE_PARALLEL_BUILD='1',PYTHONUNBUFFERED='1',CUDA_HOME=str(cuda),SGLANG_CACHE_DIR=str(Path.home()/'.cache/okc-marlin-tiles'),SGLANG_JIT_CACHE_DIR=str(Path.home()/'.cache/okc-marlin-tiles/jit'))
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+str(cuda/'bin')+os.pathsep+os.environ.get('PATH','')
os.environ['LD_LIBRARY_PATH']=str(cuda/'lib64')+':/usr/lib/wsl/lib:'+os.environ.get('LD_LIBRARY_PATH','')
os.environ['CPATH']=str(headers/'python3.12')+':'+str(headers)+':'+os.environ.get('CPATH','')
command=['/usr/bin/timeout','--signal=TERM','--kill-after=15s','240s',sys.executable,str(helper),'--workspace',str(a.workspace),'--out',str(a.out)]
receipt={'status':'prepared','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'source_hashes':tracked,'nvcc_sha256':sha(nvcc),'exit_code':None,'timeout_seconds':240}
path=a.out/'run_receipt.json'
def save():
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(receipt,indent=2)+'\n');tmp.replace(path)
save();start=time.monotonic();child=subprocess.Popen(command,start_new_session=True);receipt.update(status='running',child_pid=child.pid);save()
try:code=child.wait(timeout=270)
except subprocess.TimeoutExpired:
    receipt['supervisor_timeout']=True;os.killpg(child.pid,signal.SIGTERM)
    try:code=child.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
receipt.update(status='completed' if code==0 else 'failed',exit_code=code,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,source_unchanged=all(sha(Path(k))==v for k,v in tracked.items()))
report=a.out/'report.json'
if report.exists():receipt['report_sha256']=sha(report);receipt['report_status']=json.loads(report.read_text()).get('status')
save();print(json.dumps(receipt),flush=True)
if not receipt['source_unchanged']:raise SystemExit('Source changed during run')
raise SystemExit(code)
