"""Bind local evidence to tested source and apply unchanged paired statistics."""
import hashlib,json,subprocess,sys
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=['prologue_package/kernel.py','prologue_package/runtime.py','chain_entry.py','chain_fixture.py','component_fixture.py','guard_probe.py']
manifest=dict(sources={f:sha(here/f) for f in files},scope='Local SM86 software FP8 guarded native18-head prologue; synthetic four-block chain uses corrected BF16 native FA3 bridge.',
 raw={p.name:sha(p) for p in [here/'chain_pilot.json',here/'component_9273511.json',here/'guard_probe.json']},
 limitations=['No H200 execution','No real targetweights or distributed transport','Wholechain pilot neutral; no repeated seed seeking favorable noise','No serving or2x claim'],
 failed_attempt='run_guard_pilot/receipt.json: debug instrumentation trapped intentional uint32 encoder wrap; retained before explicit modulo fix')
path=here/'experiment_manifest.json';path.write_text(json.dumps(manifest,indent=2));variant=sha(path)
raw=json.loads((here/'chain_pilot.json').read_text())
raw.update(candidate_sha256=variant,variant_identity='native18_guarded_software_modulo_debug+unchanged_attention_moe')
(here/'chain_pilot_canonical.json').write_text(json.dumps(raw,indent=2))
d=json.loads((here/'component_9273511.json').read_text())
exact=d['cast_exact'] and all(all(all(check.values()) for check in iteration.values()) for c in d['cases'] for iteration in c['checks'])
c=dict(agent='decode',candidate='guarded_native18_prologue_component',scope='QK norm/RoPE+FP8 QKVcast+sparsecachewrite only; SM86 software conversion; excludes attention and serving',
 baseline_sha256=raw['baseline_sha256'],candidate_sha256=variant,variant_identity='native18_guarded_software_modulo_debug',seed=d['seed'],complete=d['complete'],active=True,correct=exact,
 bench=dict(instances=1,shapes={f"M{x['M']}_{x['sliding']}":dict(weight=1.,times=dict(baseline=x['times']['native'],candidate=x['times']['candidate'],control=x['times']['control'])) for x in d['cases']}))
(here/'component_pilot_canonical.json').write_text(json.dumps(c,indent=2))
for name in ('chain','component'):
 subprocess.run([sys.executable,str(root/'research_2026-11-18_rl/score_trials.py'),str(here/f'{name}_pilot_canonical.json'),'--output',str(here/f'{name}_reward.json')],check=True)
