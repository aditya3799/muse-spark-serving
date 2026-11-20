"""Paired bootstrap against both identical parent arms, without assuming an H200 clock."""
from pathlib import Path
import json
import math
import random
from statistics import median

here=Path(__file__).resolve().parent
def describe(v,rng):
    boot=sorted(median(rng.choices(v,k=len(v))) for _ in range(5000))
    return dict(median=median(v),ci95=[boot[124],boot[4874]])
out=[]
for seed in (7219,8842):
    run=json.loads((here/f"gpu_stage_seed{seed}.json").read_text())
    rng=random.Random(seed)
    for c in run['cases']:
        a,b,q=(c['wall_us'][k] for k in ('baseline','candidate','control'))
        v=[100.*math.log(math.sqrt(x*z)/y) for x,y,z in zip(a,b,q)]
        dup=[100.*math.log(z/x) for x,z in zip(a,q)]
        delta=[(x+z)/2-y for x,y,z in zip(a,b,q)]
        out.append(dict(seed=seed,b=c['bs'],centered_log_reward=describe(v,rng),
                        duplicate_control_log=describe(dup,rng),saved_wall_us=describe(delta,rng),
                        median_wall_us={k:median(vv) for k,vv in c['wall_us'].items()}))
(here/"gpu_stage_signed.json").write_text(json.dumps(out,indent=2))
for row in out:
    print(row['seed'],row['b'],tuple(round(v,2) for v in row['saved_wall_us']['ci95']),round(row['centered_log_reward']['median'],2),tuple(round(v,2) for v in row['centered_log_reward']['ci95']))
