"""Bind final source and validation evidence without promoting a serving result."""
import hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
sources=json.loads((here/'output_n_package/source_manifest.json').read_text())
for name,expected in sources.items():assert digest(here/'output_n_package'/name)==expected,name
probe=json.loads((here/'output_n_package_probe_v2.json').read_text())
gloo=json.loads((here/'output_n_gloo_v2.json').read_text())
policy=json.loads((here/'output_n_policy_gate.json').read_text())
assert probe['correct'] and gloo['correct'] and policy['correct']
for name,sha in probe['package_sources'].items():
 if name=='runtime_projection.py':
  assert sha==policy['tested_gpu_runtime_sha256'] and sources[name]==policy['final_runtime_sha256']
 else:assert sha==sources[name],name
evidence=['output_n_package_probe_v2.json','output_n_gloo_v2.json','output_n_policy_gate.json',
          'output_n_experiment_manifest.json','output_n_reward.json']
doc=dict(package_source_manifest='output_n_package/source_manifest.json',
 package_source_manifest_sha256=digest(here/'output_n_package/source_manifest.json'),
 evidence={name:digest(here/name) for name in evidence},
 local_package_validation_passed=True,h200_validation=False,nccl_performance_validated=False,
 serving_speedup_claim=None,scored_submission_recommended=False,
 scope='Real checkpoint local math+graph/lifetime+fallback; real Gloo control with mocked math. Ideal-transport component statistics are a distinct artifact, not package/serving throughput.',
 final_runtime_delta=policy['delta'],
 primary_policy_source=dict(url=policy['primary_source'],snapshot_sha256=digest(here/'sources/pytorch_CUDABlas.cpp')),
 builder='python research_2026-11-19/prefill/build_integration.py --enable-n --name integration_n',
 baseline_sha256='8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b')
(here/'PACKAGE_VALIDATION.json').write_text(json.dumps(doc,indent=2));print(json.dumps(doc,indent=2))
