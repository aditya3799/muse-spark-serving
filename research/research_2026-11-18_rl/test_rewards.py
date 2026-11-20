"""Deterministic regression controls for the scoring and confirmation rules."""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from score_trials import BASE, PLUGIN, score

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kar_stats", PLUGIN)
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)


def fixture(ratios, control=1, correct=True, active=True, seed=123):
    shapes = {}
    for index, factor in enumerate(ratios):
        baseline = [100 + ((j + seed) % 7) for j in range(16)]
        shapes[str(index)] = dict(weight=1, times=dict(baseline=baseline,
            candidate=[v / factor for v in baseline], control=[v / control for v in baseline]))
    return dict(agent="test", candidate="synthetic", scope="fixture", seed=seed,
                baseline_sha256=BASE, correct=correct, active=active, bench=dict(instances=1, shapes=shapes))


def main():
    controls = [
        (fixture([1.0]), "NO-GAIN", 0), (fixture([1.2]), "KEEP", 1),
        (fixture([.8]), "SLOWER", -1), (fixture([2, .8]), "MIXED", -1),
        (fixture([1.4], control=.8), "NOISY", 0),
        (fixture([2], correct=False), "INCORRECT", -1),
        (fixture([2], active=False), "UNMEASURED", None),
    ]
    with tempfile.TemporaryDirectory(dir=here) as tmp:
        p=Path(tmp)/"trial.json"
        for payload, verdict, sign in controls:
            p.write_text(json.dumps(payload))
            r = score(p, stats)
            assert r["decision"] == verdict
            assert (None if r["reward"] is None else (r["reward"]>0)-(r["reward"]<0)) == sign
        p.write_text(json.dumps(fixture([1.2])))
        held=Path(tmp)/"other.json"
        held.write_text(json.dumps(fixture([1.19], seed=9327)))
        cmd=[sys.executable, "-B", str(here/"score_trials.py"), str(p), "--holdout", str(held)]
        result=json.loads(subprocess.check_output(cmd))
        assert result["local_keep"]
        stress=Path(tmp)/"stress.json"
        stress.write_text(json.dumps(dict(baseline_sha256=BASE,candidate="synthetic",active=True,correct=False)))
        score_file=Path(tmp)/"result.json"
        veto=json.loads(subprocess.check_output(cmd+["--validation",str(stress),"--output",str(score_file)]))
        assert not veto["local_keep"] and json.loads(score_file.read_text())["correctness_penalty"] == -100
        partial=fixture([1.2,1.1],seed=411)
        partial["complete"]=False
        held.write_text(json.dumps(partial))
        incomplete=json.loads(subprocess.check_output(cmd))
        assert not incomplete["local_keep"] and incomplete["runs"][-1]["decision"]=="INCOMPLETE"
        assert incomplete["runs"][-1]["reward"] is None
        held.write_text(p.read_text())
        assert subprocess.run(cmd, capture_output=True).returncode != 0
    print("Eleven reward and confirmation controls passed")


if __name__ == "__main__":
    main()