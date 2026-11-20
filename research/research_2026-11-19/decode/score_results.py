"""Use the saved reward comparator without changing historical receipts."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "research_2026-11-18_rl/swa_schedule"
scorer = ROOT / "research_2026-11-18_rl/score_trials.py"

def convert(path, arm, prefix):
    data = json.loads(path.read_text())
    report = dict(agent="decode_research", candidate="Native 62-layer attention: " + arm,
                  scope=data["scope"], baseline_sha256=data["baseline_sha256"], seed=data["seed"],
                  correct=data["correct"], complete=data["complete"], active=data["active"],
                  candidate_sha256=hashlib.sha256((OLD / "native_stack.py").read_bytes()).hexdigest(),
                  variant_identity=arm, original_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  bench=dict(instances=1, shapes={}))
    for key, row in data["cases"].items():
        report["bench"]["shapes"][key] = dict(weight=1, times={name: row["times"][source]
            for name, source in (("baseline", "baseline"), ("control", "control"), ("candidate", arm))})
    target = HERE / f"{prefix}_{data['seed']}_{arm}.json"
    target.write_text(json.dumps(report, indent=2))
    return target

def score(pilot, holdout, out):
    subprocess.run([sys.executable, str(scorer), str(pilot), "--holdout", str(holdout),
                    "--output", str(HERE / out)], check=True)

pilot = OLD / "native_stack_768103.json"
held = HERE / "native_1159837/result.json"
for arm in ("schedule", "direct_schedule", "private_schedule"):
    score(convert(pilot, arm, "native"), convert(held, arm, "native"), f"native_{arm}_reward.json")
score(HERE / "chain_760183/result.json", HERE / "chain_1387291/result.json", "chain_reward.json")
