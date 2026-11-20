import hashlib,json,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
for name in ('pilot','holdout'):
    p=H/(name+'.json');r=json.loads(p.read_text())
    correct=all(c['random']==c['changed_graph_overflow']==[0]*4 for c in r['cases']) and all(x['byte_diffs']==0 for x in r['reference'])
    bench=dict(instances=1,shapes={f"b{c['batch']}_q{c['q']}_rope{int(c['rope'])}":dict(weight=1,times=c['times']) for c in r['cases']})
    out=dict(agent='prefill',candidate='Fuse target FA3 QK/RoPE, FP8 Q/K/V and native cache append',scope=r['scope'],correct=correct,active=True,seed=r['seed'],candidate_sha256=r['candidate_sha256'],baseline_sha256=r['baseline_sha256'],raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bench=bench)
    (H/(name+'_canonical.json')).write_text(json.dumps(out,indent=2))
scorer=H.parents[1]/'research_2026-11-18_rl/score_trials.py'
subprocess.run([sys.executable,str(scorer),str(H/'pilot_canonical.json'),'--holdout',str(H/'holdout_canonical.json'),'--output',str(H/'entry_reward.json')],check=True)

