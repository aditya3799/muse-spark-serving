"""Score matching partial-acceptance pilot/holdout scopes; preserve raw receipts."""
import json,subprocess,sys,hashlib
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1]
raw=here/'output_shard_9317861/result.json';doc=json.loads(raw.read_text())
pilot_raw=here/'output_shard_5791831/result.json';pilot=json.loads(pilot_raw.read_text())
dependencies=['research_2026-11-18_lt_audit/sweep.cc','research_2026-11-18_lt_audit/sweep.so',
 'research_2026-11-18_rl/lt_probe.cc','research_2026-11-18_lt_audit/native_abi.py',
 'research_2026-11-18_lt_audit/runtime_projection.py','research_2026-11-18_lt_audit/runtime_lt.cc',
 'research_2026-11-18_lt_audit/coupled_full.py','research_2026-11-18_lt_audit/real_coupled.py',
 'research_2026-11-18_rl_dflash/joint_bench.py']
manifest=dict(kind='measured ideal-transport experiment bundle, NOT deployable package identity',
 inputs=[dict(path=str(p.relative_to(root)),receipt_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
  tested_script_sha256=d['source_sha256']) for p,d in [(pilot_raw,pilot),(raw,doc)]],
 dependencies={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in dependencies},
 frozen_algorithms={k:v['plan']['selected'] for k,v in pilot['details'].items()},
 excludes='NCCL, target/draft transformer, packaged router; package must have its own validation receipt')
manifest_path=here/'output_n_experiment_manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2))
identity=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
for d in (pilot,doc):
 d['candidate_sha256']=identity;d['variant_identity']='real-weight full-K output-N frozen-parent Lt experiment; ideal transport only'
pilot_canonical=pilot_raw.with_name('canonical_bound.json');pilot_canonical.write_text(json.dumps(pilot,indent=2))
doc['derivation']={'raw':str(raw),'changes':'Restrict timing shapes to partial acceptance for matching pilot; full-acceptance evidence remains in raw receipt.'}
doc['bench']['shapes']={k[:-8]:v for k,v in doc['bench']['shapes'].items() if k.endswith('_partial')}
derived=here/'output_shard_9317861/partial_canonical.json';derived.write_text(json.dumps(doc,indent=2))
reward=here/'output_n_reward.json';old=here/'output_n_reward_unbound.json'
if reward.exists() and not old.exists():old.write_bytes(reward.read_bytes())
subprocess.run([sys.executable,str(root/'research_2026-11-18_rl/score_trials.py'),str(pilot_canonical),
 '--holdout',str(derived),'--output',str(here/'output_n_reward.json')],check=True)
