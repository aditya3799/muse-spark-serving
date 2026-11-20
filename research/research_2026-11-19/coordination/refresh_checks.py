"""Read existing check statuses once; never launches or submits a job."""
import concurrent.futures, datetime, json, pathlib, subprocess
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[1]
state=json.loads((HERE/'SESSION_STATE.json').read_text(encoding='utf-8-sig'))
keys={'old_pending_check':'pending_hopper_status.json','cache_check':'cache_check_status.json','projection_check':'projection_check_status.json','moe_check':'moe_check_status.json'}

def fetch(item):
 key,filename=item
 job=state['remote'].get(key)
 if not job:return {'name':key,'status':'not_launched'}
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 try:
  result=subprocess.run(['/mnt/d/OpenKernelChallenge/bin/okc','serve','status',job,'--json'],cwd=ROOT,capture_output=True,text=True,timeout=90)
  if result.returncode:raise RuntimeError(result.stderr[-1000:])
  data=json.loads(result.stdout)
  if data.get('id')!=job:raise ValueError('returned job identity differs')
  (HERE/filename).write_text(json.dumps(data,indent=2)+'\n')
  return {'name':key,'job':job,'checked_utc':now,'status':data.get('status'),'result_present':data.get('result') is not None}
 except Exception as exc:
  return {'name':key,'job':job,'checked_utc':now,'error':str(exc)}

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 rows=list(pool.map(fetch,keys.items()))
with (HERE/'check_status_history.jsonl').open('a') as f:
 for row in rows:f.write(json.dumps(row)+'\n')
print(json.dumps(rows,indent=2))
raise SystemExit(1 if any('error' in row for row in rows) else 0)
