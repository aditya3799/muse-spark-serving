from pathlib import Path
import hashlib,json,datetime,shutil
here=Path(__file__).resolve().parent;out=here/'h200_plan_frozen';out.mkdir(exist_ok=True)
assert not (out/'manifest.json').exists(),'Frozen manifest already exists'
names=['evaluate_h200_receipt.py','test_h200_receipt.py','h200_evaluator_contracts.json','H200_VALIDATION_PLAN.txt','H200_EVALUATOR_REVIEW.txt','build_h200_template.py','h200_receipt_template.json','h200_current_readiness.json','h200_current_decision.json']
rows=[]
for name in names:
 src=here/name;dst=out/name;shutil.copyfile(src,dst)
 rows.append(dict(file=name,bytes=dst.stat().st_size,sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
r=dict(frozen_at_utc=datetime.datetime.utcnow().isoformat()+'Z',files=rows,disabled=True,launch_capability=False,no_gpu_or_network_or_subprocess=True,synthetic_tests_are_not_hardware_evidence=True,scored_submission_authorized=False)
(out/'manifest.json').write_text(json.dumps(r,indent=2));print(json.dumps(dict(manifest_sha256=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),files=len(rows),bytes=sum(x['bytes'] for x in rows)),indent=2))
