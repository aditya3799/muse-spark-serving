"""Bounded single child, original exit receipt, no GPU imports in supervisor."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
from metadata_probe import PINS

P = Path(__file__).resolve().parent
pa = argparse.ArgumentParser()
pa.add_argument('--out', type=Path, required=True)
pa.add_argument('--seconds', type=int, default=240)
args = pa.parse_args()
assert 30 <= args.seconds <= 300
args.out.mkdir(parents=True, exist_ok=True)
script = P / 'metadata_probe.py'
os.environ.update(MAX_JOBS='1', PYTHONUNBUFFERED='1')
command = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=15s', str(args.seconds) + 's',
           sys.executable, str(script), '--out', str(args.out)]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
receipt = dict(status='started', started_utc_seconds=time.time(), supervisor_pid=os.getpid(),
               command=command, script_sha256=sha(script), launcher_sha256=sha(Path(__file__)))
snap = args.out / 'source_snapshots'
snap.mkdir(exist_ok=True)
for p in (script, Path(__file__)):
    (snap / p.name).write_bytes(p.read_bytes())
receipt['source_snapshots'] = {}
for index, (source, digest) in enumerate(PINS.items()):
    assert sha(source) == digest, source
    target = snap / ('input_%d_%s' % (index, source.name))
    target.write_bytes(source.read_bytes())
    receipt['source_snapshots'][str(source)] = dict(file=target.name, sha256=sha(target))
def save():
    tmp = args.out / 'run_receipt.tmp'
    tmp.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    tmp.replace(args.out / 'run_receipt.json')
save()
with (args.out / 'stdout.log').open('w') as stdout, (args.out / 'stderr.log').open('w') as stderr:
    child = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
    receipt['child_pid'] = child.pid
    save()
    print(json.dumps(receipt), flush=True)
    code = child.wait()
receipt.update(status='exited', exit_code=code, finished_utc_seconds=time.time())
result = args.out / 'results.json'
if result.exists(): receipt['report_sha256'] = sha(result)
save()
print(json.dumps(receipt), flush=True)
sys.exit(code)
