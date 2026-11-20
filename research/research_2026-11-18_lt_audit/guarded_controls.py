"""Audit the native-control bypass cells without changing the declared promotion verdict."""
import importlib.util
import json
from pathlib import Path
import statistics

here=Path(__file__).resolve().parent
path=here.parents[1]/"scratch/kar-serving-research/kar/evaluate/stats.py"
spec=importlib.util.spec_from_file_location("kar_local_stats",path)
stats=importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)
official=json.loads((here/"guarded_reward.json").read_text())
out={}
for stage,run in zip(("pilot","holdout"),official["runs"]):
    raw=json.loads((here/f"guarded_{stage}.json").read_text())
    secondary=stats.compare(raw["bench"],["candidate"],["control"])
    rows={}
    def ratio(z):
        return {"gain":1+z["effect"],"ci":[1+x for x in z["ci"]]}
    for name,cell in raw["bench"]["shapes"].items():
        rows[name]={"candidate_vs_baseline":ratio(run["shapes"][name]),"control_vs_baseline":ratio(run["noise"]["shapes"][name]),
                    "candidate_vs_control":ratio(secondary["shapes"][name]),
                    "median_wall_ms":{k:statistics.median(v)/1000 for k,v in cell["times"].items()}}
    out[stage]={"official_decision":run["decision"],"shapes":rows}
(here/"guarded_aa.json").write_text(json.dumps(out,indent=2)+"\n")
for stage,d in out.items():
    for n,r in d["shapes"].items():
        if n.startswith(("B1_","B4_")):
            print(stage,n,r["candidate_vs_control"])

