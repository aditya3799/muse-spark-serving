"""One bounded stability gate; no relaxed threshold, one compiler exec only."""
import json,os,sys,time
from pathlib import Path
here=Path(__file__).resolve().parent;limit=2.3*1024**3
old=here/'compile_preflight_failure.json';saved=here/'compile_preflight_initial.json'
if old.exists() and not saved.exists():saved.write_bytes(old.read_bytes())
start=time.monotonic();rows=[];consecutive=0
while time.monotonic()-start<=55:
 available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
 consecutive=consecutive+1 if available>=limit else 0
 rows.append(dict(elapsed=time.monotonic()-start,available=available,consecutive=consecutive))
 (here/'stable_preflight.json').write_text(json.dumps(dict(required=limit,observations=rows,compiler_started=consecutive>=3),indent=2))
 if consecutive>=3:
  print('stable headroom; one approved offline compiler exec',flush=True)
  os.execv(sys.executable,[sys.executable,str(here/'compile_offline.py')])
 time.sleep(10)
print('BLOCKED unchanged memory threshold; no compiler started')
sys.exit(2)
