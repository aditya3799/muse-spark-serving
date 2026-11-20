import hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent
files=['deferred_sync.py','build_deferred_sync.py','deferred_sync_gloo.py','run_deferred_sync_final_cpu.py','deferred_sync_host_cost.py','deferred_sync_host_cost.json','DEFERRED_SYNC_READOUT.txt','DFLASH_SYNC_AUDIT.txt','deferred_sync_final_gloo_tp2.json','deferred_sync_final_gloo_tp4.json','deferred_sync_final_gloo_memory.json','deferred_sync_integration/delta.diff','deferred_sync_integration/candidate.diff','deferred_sync_integration/validation.json','sync_sources/manifest.json','latest_dflash/manifest.json']
for world in (2,4):assert json.loads((here/f'deferred_sync_final_gloo_tp{world}.json').read_text())['passed']
runtime=hashlib.sha256((here/'deferred_sync.py').read_bytes()).hexdigest()
assert json.loads((here/'deferred_sync_host_cost.json').read_text())['source_sha256']==runtime
doc=dict(frozen=True,applied=False,default_enabled=False,local_value_equivalence=True,net_performance_measured=False,promoted=False,
 candidate_sha256=hashlib.sha256((here/'deferred_sync_integration/candidate.diff').read_bytes()).hexdigest(),
 files={name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in files},
 remaining=['actual native GPU softcap/accept integration','NCCL8-rank ordering/graph lifecycle','complete server startup','H200 critical-path time including host bookkeeping'])
(here/'deferred_sync_freeze_manifest.json').write_text(json.dumps(doc,indent=2));print(json.dumps({k:v for k,v in doc.items() if k!='files'},indent=2))
