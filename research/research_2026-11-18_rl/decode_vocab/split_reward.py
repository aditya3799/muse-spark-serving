"""Score local target selection and full projection as distinct scopes."""
import json
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
raw_paths = [here / "pilot.json", here / "holdout.json"]
for prefix, label in (("head","FP32 target output projection, output scaling and greedy token selection on a single TP8 shard."), ("vocab","Scaling and greedy selection only on one target vocabulary shard.")):
    paths = []
    for path in raw_paths:
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        scoped = dict(raw, scope=label)
        scoped["bench"] = dict(raw["bench"], shapes={k:v for k,v in raw["bench"]["shapes"].items() if k.startswith(prefix)})
        scoped["raw_trial"] = path.name
        if not scoped["bench"]["shapes"]:
            continue
        output = path.with_stem(path.stem+"_"+prefix)
        output.write_text(json.dumps(scoped,indent=2))
        paths.append(output)
    if paths:
        cmd = [sys.executable, str(here.parent/"score_trials.py"),str(paths[0]),"--output",str(here/(prefix+"_reward.json"))]
        if len(paths)>1:
            cmd.extend(["--holdout",str(paths[1])])
        subprocess.run(cmd,check=True)