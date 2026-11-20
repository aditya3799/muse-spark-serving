from pathlib import Path
import hashlib,json,shutil,datetime
here=Path(__file__).resolve().parent;out=here/'finite_frozen';out.mkdir(exist_ok=True)
names=['hopper_expert_register_finite.cu','compile_wgs_finite.py','finite_scale_state.py','finite_scale_scan.py','test_finite_scale_state.py','test_finite_scale_scan.py','finite_scale_state_cpu.json','finite_scale_scan_gpu.json','finite_wgs_source_plan.json','wgs_finite_compile.json','wgs_finite_2_compile.log','register_wgs_disassembly.json','register_dequant.cuh','register_dequant_gpu.json','register_dequant_cpu.json','register_dequant_compile.json','register_wgs_source_audit.json']
rows=[]
for name in names:
 src=here/name;dst=out/name
 if dst.exists():assert dst.read_bytes()==src.read_bytes(),('freeze already exists with differing source',name)
 else:shutil.copyfile(src,dst)
 rows.append(dict(file=name,bytes=dst.stat().st_size,sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
report=dict(frozen_at_utc=datetime.datetime.utcnow().isoformat()+'Z',files=rows,source_only_research=True,runtime_enabled=False,scored_submission_allowed=False,remaining_gates=['Independent certificate/graph lifecycle review','Actual H200 full routed-kernel correctness and controlled phase performance','Full model output quality and serving peak-memory/latency gates'],graph_contract='Weights immutable throughout captured graph lifetime; mutation/reload retires graphs before reuse. Python version checks only protect eager/pre-capture dispatch.',binary_reference=dict(file='../wgs_finite_2_sm90a.cubin',sha256=hashlib.sha256((here/'wgs_finite_2_sm90a.cubin').read_bytes()).hexdigest()),build_scope='Offline SM90a cubin only; local coefficient numerical execution was SM86 with distinct lowering. No Hopper runtime proof.')
(out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(files=len(rows),bytes=sum(x['bytes'] for x in rows),manifest_sha256=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),source_sha256=rows[0]['sha256']),indent=2))
