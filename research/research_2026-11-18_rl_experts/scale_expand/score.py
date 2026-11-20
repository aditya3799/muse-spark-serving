"""Select a positive, controlled pilot winner and evaluate only its fresh holdout."""
import argparse,hashlib,json,math,subprocess,sys
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1]
parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("pilot","holdout"));args=parser.parse_args()
scorer=root/"research_2026-11-18_rl/score_trials.py";doc=json.loads((here/(args.mode+".json")).read_text())
assert doc["complete"] and all(all(x["exact"] for x in r["patterns"]) for r in doc["stage_parity"])
ranks=[]
for variant in doc["variants"]:
    variant_id=hashlib.sha256((here/"policy.py").read_bytes()+variant.encode()).hexdigest()
    shapes={}
    for name,r in doc["benchmarks"].items():
        assert all(r["parity"])
        shapes[name]=dict(weight=1,times={"baseline":r["times"]["parent"],"candidate":r["times"][variant],"control":r["times"]["control"]},
                          parity=r["parity"],order=r["order"])
    raw=dict(agent="experts",candidate="native_bf16_group_scale_"+variant,scope="Rank-local FFN on official busiest-rank fixtures",
        seed=doc["seed"],baseline_sha256=doc["parent_sha256"],candidate_sha256=variant_id,
        correct=True,active=any(r["role"]==variant for r in doc["kernels"]),complete=True,bench=dict(instances=1,shapes=shapes))
    dest=here/(args.mode+"_"+variant+".json");dest.write_text(json.dumps(raw,indent=2))
    out=here/(args.mode+"_"+variant+"_reward.json")
    cmd=[sys.executable,str(scorer),str(here/("pilot_"+variant+".json"))]
    if args.mode=="holdout":cmd+=["--holdout",str(dest)]
    subprocess.run(cmd+["--output",str(out)],check=True)
    row=json.loads(out.read_text())["runs"][-1]
    ranks.append((row["reward"],row["decision"],variant,row["gain"],row["ci"]))
print(sorted(ranks,reverse=True),flush=True)
(here/(args.mode+"_ranking.json")).write_text(json.dumps(sorted(ranks,reverse=True),indent=2)+"\n")

