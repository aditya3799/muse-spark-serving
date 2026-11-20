"""Supervise one bounded row-norm probe and durably record its original exit."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,signal,subprocess,sys,time
os.environ['ROW_NORM_BOUND']='240'
os.environ['MAX_JOBS']='1'
base=Path(__file__).resolve().parent;helper=base/'probe_row_local_norm.py'
args=sys.argv[1:]
if '--out' not in args:raise SystemExit('--out is required')
out=Path(args[args.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
receipt_path=out/'run_receipt.json'
if receipt_path.exists():raise SystemExit('Use a fresh output directory; existing receipt preserved')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
snapshot=out/'source_snapshot';snapshot.mkdir()
manifest=base/'manifest.json';manifest_data=json.loads(manifest.read_text())
baseline=base.parents[2]/'coordination/round5_humming_adaptive_fused/candidate_sources'
sglang=Path(args[args.index('--sglang-root')+1])
copies=[(helper,Path(helper.name)),(Path(__file__),Path(__file__).name),(manifest,'manifest.json')]
for name in manifest_data['sources']:
 copies.append((base/'candidate_sources'/name,Path('candidate')/name))
 copies.append((baseline/name,Path('baseline')/name))
copies.append((sglang/'python/sglang/srt/layers/muse_spark/muse_spark_layers.py',Path('norm_source/muse_spark_layers.py')))
tracked_sources={}
for original,relative in copies:
 destination=snapshot/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,destination);tracked_sources[str(original)]=sha(original)
command=['/usr/bin/timeout','--signal=TERM','--kill-after=15s','240s',sys.executable,str(helper),*args]
receipt={'status':'prepared','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'launcher_argv':sys.argv,'helper_sha256':sha(helper),'launcher_sha256':sha(Path(__file__)),'source_snapshots':{str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file()},'tracked_original_sources':tracked_sources,'exit_code':None,'scope':'Single-GPU arithmetic and normalization component timing only; no real collectives or serving proof'}
def save():
 temp=receipt_path.with_suffix('.tmp');temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(receipt_path)
save();started=time.monotonic();child=subprocess.Popen(command,start_new_session=True);receipt.update(status='running',child_pid=child.pid,process_group=child.pid);save()
try:
 code=child.wait(timeout=270)
except subprocess.TimeoutExpired:
 receipt['supervisor_timeout']=True
 os.killpg(child.pid,signal.SIGTERM)
 try:code=child.wait(timeout=15)
 except subprocess.TimeoutExpired:
  os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
receipt.update(status='completed' if code==0 else 'failed',exit_code=code,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
receipt['source_unchanged']=all(sha(Path(name))==digest for name,digest in tracked_sources.items())
report_path=out/'report.json'
if report_path.exists():
 receipt['report_sha256']=sha(report_path)
 report=json.loads(report_path.read_text());receipt['report_status']=report.get('status');receipt['compile_seconds']=report.get('compile_seconds')
 if (out/'kernel.sm90a.cubin').exists():receipt['cubin_sha256']=sha(out/'kernel.sm90a.cubin')
save();print(json.dumps({'status':receipt['status'],'exit_code':code,'receipt':str(receipt_path),'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)
if not receipt['source_unchanged']:raise SystemExit('Source changed during run')
raise SystemExit(code)