"""Score these CPU scheduler stage receipts with the pinned local plugin statistics."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

here=Path(__file__).resolve().parent
plugin=here.parents[2]/"scratch/kar-serving-research/kar/evaluate/stats.py"
spec=importlib.util.spec_from_file_location("kar_stats",plugin)
stats=importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)
manifest=json.loads((here/"manifest.json").read_text())
out=dict(scope="CPU scheduler receive poll plus a four-int64 Gloo MIN; no H200 estimate",
         plugin_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest(),runs={})
for path in sorted(here.glob("cpu_tp8_seed*.json")):
    receipt=json.loads(path.read_text())
    bench={"instances":1,"shapes":{}}
    for name,case in receipt["cases"].items():
        bench["shapes"][name]={"weight":1,"times":case["times_us_per_poll_including_min"]}
    canonical=dict(agent="experts",candidate="same-host TP ring for empty ingress polls",
                   variant_identity="original Gloo payload on every nonempty poll",
                   candidate_sha256=manifest["candidate_sha256"],scope=out["scope"],
                   baseline_sha256=manifest["parent_sha256"],correct=True,active=True,
                   seed=receipt["seed"],bench=bench)
    (here/("canonical_cpu_%s.json"%receipt["seed"])).write_text(json.dumps(canonical,indent=2))
    rows={}
    metrics={"equal_weight_stage_diagnostic":bench,**{
        name:dict(instances=1,shapes={name:shape}) for name,shape in bench["shapes"].items()}}
    for k,metric in metrics.items():
        a=stats.compare(metric,["candidate"],["baseline"])
        null=stats.compare(metric,["control"],["baseline"])
        decision,why=stats.decide(a,noise=null)
        gain=1+a["effect"]
        rows[k]=dict(adjacent_ab_ba_pairs=receipt["rounds"]//2,ratio=gain,
                     ci_95=[1+s for s in a["ci"]],AA_ratio=1+null["effect"],
                     AA_ci_95=[1+s for s in null["ci"]],signed_log_reward=100*math.log(gain),
                     decision=decision,reason=why,
                     noise_gated_reward=100*math.log(gain) if decision in ("KEEP","SLOWER") else 0.)
    out["runs"][str(receipt["seed"])]=rows
(here/"cpu_signed_rewards.json").write_text(json.dumps(out,indent=2))
print([(seed,{k:[x["decision"],round(x["signed_log_reward"],2),[round(v,3) for v in x["ci_95"]]] for k,x in row.items()}) for seed,row in out["runs"].items()])
