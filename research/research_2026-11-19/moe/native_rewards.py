"""Separate complete rank-local expert experiments from whole-model performance."""
from pathlib import Path
import argparse,json,math,random
from statistics import median

here=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument("mode",choices=["fallback","direct","vector","scales"]);p.add_argument("seeds",nargs="+",type=int)
args=p.parse_args()
for seed in args.seeds:
    x=json.loads((here/f"{args.mode}_native_{seed}.json").read_text())
    assert x["complete"] and x["correct"]
    shapes={};summary=[]
    for r in x['rows']:
        times=r['times_us'];a,b,c=(times[k] for k in ('parent','candidate','control'))
        pair=[100*math.log(math.sqrt(first*dup)/trial) for first,trial,dup in zip(a,b,c)]
        rng=random.Random(seed+r['m'])
        bs=sorted(median(rng.choices(pair,k=len(pair))) for _ in range(5000))
        summary.append(dict(m=r['m'],pattern=r['route'],parent_us=median(a),candidate_us=median(b),control_us=median(c),
                            signed_log_reward=median(pair),bootstrap95_log=[bs[124],bs[4874]],
                            matching_all_live_and_final_bits=True,observed_active=r['live']))
        shapes['m%d_%s'%(r['m'],r['route'])]=dict(weight=1,times=dict(baseline=a,candidate=b,control=c))
    active=bool(x.get("python_dispatch_counts",{}).get("w13_"+args.mode,0)) if args.mode!="fallback" else True
    out=dict(agent="experts",candidate=("original shared Humming coefficients + canonical scale-word native loader" if args.mode=="scales" else "exclusive CUTLASS FP4 order -> "+args.mode+" native Humming loader"),
             scope="rank-local Humming W13+SiLU+W2+finalizer on RTX3050; controlled logical routes",
             baseline_sha256=x['parent'],active=active,correct=True,seed=seed,
             bench=dict(instances=1,shapes=shapes),summary=summary)
    (here/f"{args.mode}_canonical_{seed}.json").write_text(json.dumps(out,indent=2))
    print(seed,[(r['m'],r['pattern'],round(math.exp(r['signed_log_reward']/100),3),
                 round(r['parent_us']),round(r['candidate_us'])) for r in summary])
