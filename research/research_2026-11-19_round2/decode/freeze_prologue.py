import hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent
names=['prologue_package/kernel.py','prologue_package/runtime.py','prologue_package/host.py','prologue_package/__init__.py',
 'integration/candidate.diff','integration/delta.diff','integration/validation.json','experiment_manifest.json','guard_probe.json',
 'host_guard_cpu.json','backend_guard_cpu.json','exceptional_probe.json','chain_reward.json','component_reward.json','sm90_compile/receipt.json']
manifest=dict(status='FROZEN_UNAPPLIED_RESEARCH_ONLY',enabled_by_default=False,remote_qualified=False,
 cumulative_patch_sha256=hashlib.sha256((here/'integration/candidate.diff').read_bytes()).hexdigest(),
 files={name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in names},
 outcome='Exact local correctness and offlineSM90compile; material wholechaingain absent. No servingpromotion.',
 gates_remaining=['SM90actual deployedTorch exactcast qualification','actual nativeHopperFP8attention and fullmodeloutput validation','realtransport/fullfrontier serving timing','normalstartup/captureintegration in completeSGLangenv'])
(here/'prologue_freeze_manifest.json').write_text(json.dumps(manifest,indent=2));print(manifest['cumulative_patch_sha256'])
