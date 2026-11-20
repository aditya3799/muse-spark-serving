from pathlib import Path
import hashlib,json,shutil,datetime
here=Path(__file__).resolve().parent;out=here/'split_readiness_frozen';out.mkdir(exist_ok=True)
names=['hopper_expert_split_ready.cu','hopper_expert_split_dependency.cu','build_split_ready.py','build_split_dependency.py','compile_wgs_split_ready.py','compile_wgs_split_dependency.py','split_ready_source_audit.json','split_dependency_source_audit.json','wgs_split_ready_compile.json','wgs_split_dependency_compile.json','wgs_split_ready_2_compile.log','wgs_split_dependency_2_compile.log','split_ready_disassembly.json','split_dependency_disassembly.json','test_split_ready_generations.py','split_ready_cpu_generations.json','audit_split_wait_dataflow.py','split_wait_dataflow.json','SPLIT_READINESS_REVIEW.txt']
assert not (out/'manifest.json').exists(),'Never overwrite a frozen manifest'
rows=[]
for name in names:
 src=here/name;dst=out/name;shutil.copyfile(src,dst)
 rows.append(dict(file=name,bytes=dst.stat().st_size,sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
r=dict(frozen_at_utc=datetime.datetime.utcnow().isoformat()+'Z',files=rows,runtime_enabled=False,scored_submission_allowed=False,no_hopper_execution=True,binaries=[dict(file='../'+n,sha256=hashlib.sha256((here/n).read_bytes()).hexdigest()) for n in ['wgs_split_ready_2_sm90a.cubin','wgs_split_dependency_2_sm90a.cubin']],finite_owner_contract='../finite_lifetime_v2/manifest.json',source_review='SPLIT_READINESS_REVIEW.txt')
(out/'manifest.json').write_text(json.dumps(r,indent=2));print(json.dumps(dict(files=len(rows),bytes=sum(x['bytes'] for x in rows),manifest_sha256=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()),indent=2))
