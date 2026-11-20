import hashlib,json,subprocess,sys
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1]
files=['fa3_prologue_kernel.py','fa3_prologue_probe.py','prepare_fa3_prologue.py','fa3_prologue_source.json']
identity={name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in files}
manifest=here/'fa3_prologue_experiment_manifest.json'
manifest.write_text(json.dumps(dict(sources=identity,scope='Source-derived native18head QK norm with software FP8 pack and piggybackV store. No H200/nativeFP8 conversion or serving integration.',
 raw_receipts={str(seed):hashlib.sha256((here/f'fa3_prologue_{seed}.json').read_bytes()).hexdigest() for seed in [7193851,1837951,2918357,8375193]},
 qualification_limits=['SM86 software conversion only','Synthetic QKV and physical pool mappings, true native normalization/store kernels','First pilot timings noisy and excluded','Failed nested graph timing capture retained','No hardware telemetry time series; exclusive slot and duplicate controls recorded','No attention kernel/fullchain or distributed test; historical wider fusion chain was neutral']),indent=2))
variant=hashlib.sha256(manifest.read_bytes()).hexdigest()
outputs=[]
for seed in (2918357,8375193):
 raw=here/f'fa3_prologue_{seed}.json';d=json.loads(raw.read_text())
 exact=d['cast_exact'] and all(all(all(check.values()) for check in iteration.values()) for c in d['cases'] for iteration in c['checks'])
 canonical=dict(agent='decode_research',candidate='FA3 native18-head prologue fusion',scope='SM86 QK norm/RoPE + BF16-to-FP8 Q/K/V casts + sparse cache scatter; software exact FP8 packing. Excludes attention, transport and serving.',
  baseline_sha256='8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b',
  candidate_sha256=variant,variant_identity='native18head+softwareFP8+piggybackV',seed=seed,complete=d['complete'],active=True,correct=exact,
  bench=dict(instances=1,shapes={f"M{c['M']}_{'swa' if c['sliding'] else 'full'}":dict(weight=1.,times={'baseline':c['times']['native'],'candidate':c['times']['candidate'],'control':c['times']['control']}) for c in d['cases']}),
  raw=str(raw),raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
 path=here/f'fa3_prologue_{seed}_canonical.json';path.write_text(json.dumps(canonical,indent=2));outputs.append(path)
subprocess.run([sys.executable,str(root/'research_2026-11-18_rl/score_trials.py'),str(outputs[0]),'--holdout',str(outputs[1]),'--output',str(here/'fa3_prologue_reward.json')],check=True)
