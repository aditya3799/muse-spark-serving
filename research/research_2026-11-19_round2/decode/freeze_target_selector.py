"""Bind the isolated integration to exact upstream helper proofs."""
import hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
folder=here/'target_selector_integration'
validation=json.loads((folder/'validation.json').read_text())
cpu=json.loads((here/'target_selector_cpu.json').read_text())
assert cpu['passed'] and validation['apply_check']
helper=validation['helper_sha256'];proofs={}
for name in ('correctness.json','timing_device_guard.json'):
 path=root/'coordination/target_argmax_postops'/name
 data=json.loads(path.read_text());assert data['source_sha256']==helper and data['complete']
 proofs[str(path.relative_to(root))]=sha(path)
files=['build_target_selector.py','validate_target_selector.py','target_selector_cpu.py','target_selector_cpu.json',
 'audit_muse_logits.py','MUSE_LOGITS_ACTIVATION.txt','muse_logits_activation/receipt.json','muse_logits_activation/muse_spark.py',
 'TARGET_SELECTOR_READOUT.txt','target_selector_integration/candidate.diff','target_selector_integration/delta.diff',
 'target_selector_integration/manifest.json','target_selector_integration/validation.json']
files += [str(p.relative_to(here)) for p in (folder/'sources').rglob('*.py')]
out=dict(frozen=True,applied=False,default_enabled=False,serving_qualified=False,scored_submission_authorized=False,
 candidate_sha256=validation['candidate_sha256'],helper_sha256=helper,
 cpu_actual_method_cases=len(cpu['cases']),cpu_fallback_controls=len(cpu['direct_fallback_controls']),
 gpu_proofs=proofs,files={name:sha(here/name) for name in files},
 performance='Captured local postops only; eager inconclusive; no whole-server claim.',
 remaining=['Actual server activation','H200 native execution','TP8 NCCL integration','Full serving frontier'])
(here/'target_selector_freeze_manifest.json').write_text(json.dumps(out,indent=2));print(out['candidate_sha256'])
