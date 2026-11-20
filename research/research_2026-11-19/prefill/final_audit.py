"""Freeze the reviewed local evidence and cumulative unscored diagnostic hash."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
folder=HERE/'integration_joint';manifest=json.loads((folder/'manifest.json').read_text())
patch=folder/'candidate.diff';added=[line[1:] for line in patch.read_text().splitlines() if line.startswith('+') and not line.startswith('+++')]
probe=json.loads((HERE/'row_package_probe.json').read_text())
row_source=folder/'sources/python/sglang/srt/speculative/dflash_projection_row.py'
assert sha(row_source)==probe['package_source_sha256']
assert sha(patch)==manifest['candidate_sha256']
assert probe['complete'] and all(probe['cases'].values())
assert json.loads((HERE/'integration_contract.json').read_text())['complete']
for world in (2,4,8):assert json.loads((HERE/f'row_adapter_cpu_tp{world}.json').read_text())['correct']
assert json.loads((HERE/'row_policy_gate.json').read_text())['strict_bf16_policy_rejected']
validation=json.loads((HERE/'integration_validation.json').read_text())
assert any(r['candidate_sha256']==sha(patch) and r['cumulative_apply_against_frozen_pin'] for r in validation['results'])
report=dict(candidate_sha256=sha(patch),parent_sha256=manifest['parent_sha256'],pin=manifest['pin'],
    candidate_bytes=patch.stat().st_size,max_added_line_chars=max(map(len,added)),added_line_count=len(added),
    under400KB=patch.stat().st_size<400000,max_added_line_under400=max(map(len,added))<=400,
    row_package_source_sha256=sha(row_source),n_package_manifest_sha256=sha(HERE.parent/'decode/output_n_package/source_manifest.json'),
    readyForUNSCORED=True,readyForSCORED=False,scope='Local implementation and controls complete; H200 TP8 precision, NCCL, memory, full model outputs and serving speed remain unmeasured.',
    activation=dict(row=True,output_n=True),checks={name:sha(HERE/name) for name in (
        'real_row_pilot.json','real_row_holdout.json','row_package_probe.json','row_adapter_cpu_tp2.json','row_adapter_cpu_tp4.json','row_adapter_cpu_tp8.json',
        'row_policy_gate.json','integration_contract.json','integration_validation.json')})
assert report['under400KB'] and report['max_added_line_under400'],report
(HERE/'FINAL_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
