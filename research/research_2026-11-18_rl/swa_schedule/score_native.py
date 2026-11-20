"""Apply the common signed comparator to each candidate in the native attention stack."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
here=Path(__file__).resolve().parent

def receipt(path,arm):
    d=json.loads(path.read_text())
    assert d["complete"] and d["correct"] and d["fully_materialized_layers"]==62
    assert all(r["matched_per_arm"]>0 for r in d["cases"].values())
    variants={"direct":"global full page-row map","private":"private batched page snapshots",
        "schedule":"reuse sliding-attention scheduler metadata","direct_schedule":"global full page-row map plus sliding scheduling",
        "private_schedule":"private page snapshots plus sliding scheduling"}
    bench={k:dict(weight=1,times={name:row["times"][src] for name,src in
        (("candidate",arm),("baseline","baseline"),("control","control"))}) for k,row in d["cases"].items()}
    doc=dict(agent="root",candidate="Native BF16 62-layer attention with "+variants[arm],scope=d["scope"],
        baseline_sha256=d["baseline_sha256"],seed=d["seed"],correct=True,complete=True,active=True,
        candidate_sha256=hashlib.sha256((here/"native_stack.py").read_bytes()).hexdigest(),variant_identity=arm,
        original_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bench=dict(instances=1,shapes=bench))
    p=here/f"native_{d['seed']}_{arm}_paired.json";p.write_text(json.dumps(doc,indent=2));return p
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("pilot",type=Path);p.add_argument("--holdout",type=Path);a=p.parse_args()
    arms=json.loads(a.pilot.read_text())["candidates"]
    for arm in arms:
        cmd=[sys.executable,"-B",str(here.parent/"score_trials.py"),str(receipt(a.pilot,arm))]
        if a.holdout:cmd+=["--holdout",str(receipt(a.holdout,arm))]
        subprocess.run([*cmd,"--output",str(here/f"native_{arm}_reward.json")],check=True)
