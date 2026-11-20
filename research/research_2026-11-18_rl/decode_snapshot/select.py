"""Select a snapshot tile on discovery and require fresh inputs before assigning a confirmed reward."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
here=Path(__file__).resolve().parent
def prepare(path,arm):
    trial=json.loads(path.read_text())
    assert trial["complete"] and trial["correct"]
    shapes={k:dict(weight=1,times={dst:v["times"][src] for dst,src in (("baseline","baseline"),("candidate",arm),("control","control"))}) for k,v in trial["shapes"].items()}
    name="Full/SWA private page snapshots "+arm
    d=dict(agent=trial["agent"],candidate=name,scope=trial["scope"],baseline_sha256=trial["baseline_sha256"],
        candidate_sha256=trial["kernel_sha256"],variant_identity=arm,active=True,correct=True,complete=True,seed=trial["seed"],
        original_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bench=dict(instances=1,shapes=shapes))
    outfile=here/f"cells_{trial['seed']}_{arm}.json"
    outfile.write_text(json.dumps(d,indent=2))
    return outfile
def run(paths,output):
    cmd=[sys.executable,"-B",str(here.parent/"score_trials.py"),str(paths[0])]
    if len(paths)==2:cmd.extend(["--holdout",str(paths[1])])
    subprocess.run([*cmd,"--output",str(output)],check=True)
    return json.loads(output.read_text())
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("pilot",type=Path);p.add_argument("--holdout",type=Path);a=p.parse_args()
    manifest=here/"selection.json"
    if a.holdout:
        sel=json.loads(manifest.read_text());assert sel["pilot_sha256"]==hashlib.sha256(a.pilot.read_bytes()).hexdigest()
        win=sel["selected"]
        run([prepare(a.pilot,win),prepare(a.holdout,win)],here/"snapshot_reward.json")
    else:
        d=json.loads(a.pilot.read_text());names=list(next(iter(d["shapes"].values()))["times"])
        scores={}
        for name in names:
            if not name.startswith("b") or name=="baseline":continue
            o=run([prepare(a.pilot,name)],here/f"analysis_{name}.json")["runs"][0]
            scores[name]={k:o[k] for k in ("ci","decision","gain","reward")}
        # Prioritize the best lower confidence bound, and break equal bounds by mean.
        win=max(scores,key=lambda k:(scores[k]["ci"][0],scores[k]["gain"]))
        manifest.write_text(json.dumps(dict(selected=win,pilot_sha256=hashlib.sha256(a.pilot.read_bytes()).hexdigest(),scores=scores),indent=2))
        print("selected",win,flush=True)
