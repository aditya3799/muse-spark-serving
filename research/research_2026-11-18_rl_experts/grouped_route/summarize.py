"""Preserve the interrupted pilot; score only fully populated paired fixtures."""
import json
import subprocess
import sys
from pathlib import Path
here=Path(__file__).resolve().parent
root=here.parents[1]
doc=json.loads((here/"split_pilot.json").read_text())
assert not doc["complete"] and doc["correct"] and doc["active"]
cases=("M128_normal","M128_skewed","M768_normal","M768_skewed")
assert all(len(doc["bench"]["shapes"][k]["parity"])==8 and all(doc["bench"]["shapes"][k]["parity"]) for k in cases)
summary={**doc,"complete":True,"bench":{"instances":1,"shapes":{k:doc["bench"]["shapes"][k] for k in cases}}}
summary["stratum"]="Post-interruption restriction to both 128- and 768-row cases that had all eight reciprocal rounds. M32 stopped after two of eight rounds during a device-wide timing excursion; retained in split_pilot.json. Exploratory only."
dest=here/"split_large_complete.json";dest.write_text(json.dumps(summary,indent=2))
scorer=root/"research_2026-11-18_rl/score_trials.py"
for path,reward in ((here/"split_pilot.json",here/"split_pilot_reward.json"),(dest,here/"split_large_reward.json")):
    subprocess.run([sys.executable,str(scorer),str(path),"--output",str(reward)],check=True)
