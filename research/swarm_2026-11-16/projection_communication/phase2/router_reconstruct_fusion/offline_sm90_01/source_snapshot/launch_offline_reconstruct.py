"""Supervise one offline specialization and durably record its original exit."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,signal,subprocess,sys,time
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PHASE2_OFFLINE_BOUND']='180'
os.environ['MAX_JOBS']='1'
base=Path(__file__).resolve().parent;helper=base/'offline_reconstruct_sm90.py'
args=sys.argv[1:]
if '--out' not in args:raise SystemExit('--out is required')
out=Path(args[args.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
receipt_path=out/'run_receipt.json'
if receipt_path.exists():raise SystemExit('Use a fresh output directory; existing receipt preserved')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
snapshot=out/'source_snapshot';snapshot.mkdir()
source_files=(helper,Path(__file__),base/'integration_sources/python/sglang/kernels/jit/csrc/muse_spark/routing.cuh',base/'integration_sources/python/sglang/kernels/ops/moe/muse_spark_routing_jit.py')
for file in source_files:shutil.copyfile(file,snapshot/file.name)
command=['/usr/bin/timeout','--signal=TERM','--kill-after=15s','180s',sys.executable,str(helper),*args]
receipt={'status':'prepared','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'launcher_argv':sys.argv,'helper_sha256':sha(helper),'launcher_sha256':sha(Path(__file__)),'source_snapshots':{p.name:sha(p) for p in snapshot.iterdir()},'exit_code':None,'scope':'Exact production kernel specialization compiled offline; not full installed FFI module build, runtime load, execution or H200 performance proof'}
def save():
 temp=receipt_path.with_suffix('.tmp');temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(receipt_path)
save();started=time.monotonic();child=subprocess.Popen(command,start_new_session=True);receipt.update(status='running',child_pid=child.pid,process_group=child.pid);save()
try:
 code=child.wait(timeout=210)
except subprocess.TimeoutExpired:
 receipt['supervisor_timeout']=True
 os.killpg(child.pid,signal.SIGTERM)
 try:code=child.wait(timeout=15)
 except subprocess.TimeoutExpired:
  os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
receipt.update(status='completed' if code==0 else 'failed',exit_code=code,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
receipt['source_unchanged']=all(sha(file)==receipt['source_snapshots'][file.name] for file in source_files)
report_path=out/'report.json'
if report_path.exists():
 receipt['report_sha256']=sha(report_path)
 report=json.loads(report_path.read_text());receipt['report_status']=report.get('status');receipt['compile_seconds']=report.get('compile_seconds')
 if (out/'kernel.sm90a.cubin').exists():receipt['cubin_sha256']=sha(out/'kernel.sm90a.cubin')
save();print(json.dumps({'status':receipt['status'],'exit_code':code,'receipt':str(receipt_path),'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)
if not receipt['source_unchanged']:raise SystemExit('Source changed during run')
raise SystemExit(code)