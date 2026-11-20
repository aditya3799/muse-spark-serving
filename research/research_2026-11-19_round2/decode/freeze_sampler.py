"""Freeze a standalone unapplied sampler patch and bind every measured source."""
import hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((here/'sampler_gpu_9817351.json').read_text())
assert sha(here/'sampler_measured_runtime.py')==old['source_sha256']['packed_sampler.py']
assert sha(here/'sampler_measured_kernel.py')==old['source_sha256']['packed_sampler_kernel.py']
for file in ['sampler_compact_gpu_1973519.json','sampler_class_probe.json']:
 data=json.loads((here/file).read_text())
 for name in ('packed_sampler.py','packed_sampler_kernel.py'):assert sha(here/name)==data['source_sha256'][name],(file,name)
names=['packed_sampler.py','packed_sampler_kernel.py','sampler_measured_runtime.py','sampler_measured_kernel.py','sampler_full_policy_v1.py',
 'sampler_integration/candidate.diff','sampler_integration/delta.diff','sampler_integration/manifest.json','sampler_integration/validation.json',
 'sampler_gpu_9817351.json','sampler_compact_gpu_1973519.json','sampler_class_probe.json','sampler_gloo_tp2.json','sampler_gloo_tp4.json',
 'sampler_compact_gloo_tp2.json','sampler_compact_gloo_tp4.json','sampler_policy_tp2.json','sampler_policy_tp4.json',
 'sampler_compact_policy_tp2.json','sampler_compact_policy_tp4.json','sampler_gloo_memory.json','sampler_policy_memory.json','sampler_compact_cpu_memory.json',
 'sampler_compact_preflight_failure.json','sampler_class_probe.py','sampler_edge_controls.py','sampler_gpu_probe.py','sampler_compact_gpu_probe.py',
 'sampler_gloo_probe.py','sampler_compact_gloo_probe.py','sampler_policy_gloo.py','sampler_compact_policy_gloo.py','build_sampler_integration.py']
manifest=dict(status='FROZEN_UNAPPLIED_RESEARCH_CANDIDATE',default_enabled=False,environment_flag='SGLANG_EXPERIMENTAL_DFLASH_PACKED_ARGMAX',
 parent_patch_sha256='8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b',
 cumulative_sha256=sha(here/'sampler_integration/candidate.diff'),incremental_sha256=sha(here/'sampler_integration/delta.diff'),
 files={name:sha(here/name) for name in names},
 inventory=dict(real_gloo_full_payload=336,real_gloo_compact_payload=336,full_policy_cases=18,compact_policy_cases=30,
  cuda_full_payload_pattern_cases=168,cuda_compact_payload_pattern_cases=168,cuda_actual_sampler_class_cases=144,cuda_exceptional_reduce_cases=30),
 timing_scope='Local packing+first-rank selection only. All transport including virtualpeer copies excluded. Native densehead unchanged and not in timing.',
 remaining_gates=['Actual complete SGLang startup/graph lifecycle','Real NCCL/allgather on intended8GPUH200 topology including datatype/protocol choice',
  'Actual draftgraph activation and reference output agreement with servingweights','Whole servingfrontier independent paired timing againstfrozenrelease'],
 no_scored_submission=True,combined_with_other_features=False)
(here/'sampler_freeze_manifest.json').write_text(json.dumps(manifest,indent=2));print(manifest['cumulative_sha256'])
