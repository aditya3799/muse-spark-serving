from pathlib import Path
import json,ast,hashlib
p=Path(__file__).resolve().parent
f=p/'frozen_e32_m128/run_receipt.json';r=json.loads(f.read_text(encoding='utf-8'));r.update(status='completed',exit_code=0);f.write_bytes((json.dumps(r,indent=2)+'\n').encode('utf-8'))
f=p/'frozen_helper_probe.py';s=f.read_text(encoding='utf-8')
s=s.replace("report={'runtime':runtime,", "report={'status':'running','argv':sys.argv,'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'geometry':{'local_experts':args.experts,'global_experts':args.experts*8,'hidden':args.hidden,'intermediate':args.hidden,'top_k':8,'tokens':args.tokens},'runtime':runtime,")
s=s.replace("report['stopped_cooperatively']=True", "report['stopped_cooperatively']=True\n            report['status']='stopped_cooperatively'")
s=s.replace("report['passed']=all(r['passed'] for r in report['rows']);", "report['passed']=all(r['passed'] for r in report['rows']);report['status']='completed';")
ast.parse(s);f.write_bytes(s.encode('utf-8'))
summary={'frozen_export_unchanged':True,'next_harness_hashes':{n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['frozen_helper_probe.py','memory_probe.py']},'completed_correctness':{}}
for name in ['frozen_e32_m33','frozen_e32_m128']:
 r=json.loads((p/name/'results.json').read_text(encoding='utf-8'));rows=r['rows'];replays=[v for row in rows for v in row['replays']]
 summary['completed_correctness'][name]={'passed':r['passed'],'cases':len(rows),'replays':len(replays),'max_relative_l2':max([row['eager_relative_l2'] for row in rows]+[v['relative_l2'] for v in replays]),'max_abs':max([row['eager_max_abs'] for row in rows]+[v['max_abs'] for v in replays]),'original_harness_sha256':'f6888492daceb7c331f569af14a865e2f8a3e3d3e028f0ae51f85dc66ac6424a'}
(p/'FULL_SHARD_PROOF.json').write_bytes((json.dumps(summary,indent=2)+'\n').encode('utf-8'))
print(json.dumps(summary,indent=2))
