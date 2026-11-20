"""Score a GPU pilot and independent confirmation against the frozen release."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
PLUGIN = ROOT.parent / "scratch/kar-serving-research/kar/evaluate/stats.py"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check_bench(bench):
    placements = bench.get("instances", 1)
    if not isinstance(placements, int) or placements < 1 or not bench["shapes"]:
        raise ValueError("invalid placement count or empty benchmark")
    for name, shape in bench["shapes"].items():
        times = shape["times"]
        if not {"baseline", "candidate"} <= times.keys() or not math.isfinite(shape["weight"]) or shape["weight"] <= 0:
            raise ValueError(f"invalid shape {name}")
        counts = {len(values) for values in times.values()}
        if len(counts) != 1 or min(counts) < 4 or min(counts) % (2 * placements):
            raise ValueError(f"{name}: need complete AB/BA pairs for each placement")
        if not all(math.isfinite(t) and t > 0 for values in times.values() for t in values):
            raise ValueError(f"{name}: invalid CUDA time")


def score(path, stats):
    data = read(path)
    if data["baseline_sha256"].lower() != BASE:
        raise ValueError("unexpected baseline hash")
    record = {k: data[k] for k in ("agent", "candidate", "scope")}
    record.update(source=str(path.resolve().relative_to(ROOT)).replace(os.sep, "/"),
                  source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    record["candidate_sha256"] = data.get("candidate_sha256")
    record["variant_identity"] = data.get("variant_identity")
    if data.get("complete") is False:
        return dict(record, decision="INCOMPLETE", reward=None)
    if data.get("active") is not True:
        return dict(record, decision="UNMEASURED", reward=None)
    if data.get("correct") is False:
        return dict(record, decision="INCORRECT", reward=-100.)
    if data.get("correct") is not True:
        return dict(record, decision="UNCHECKED", reward=None)
    bench = data["bench"]
    check_bench(bench)
    cmp = stats.compare(bench, ["candidate"], ["baseline"])
    controlled = all("control" in s["times"] for s in bench["shapes"].values())
    noise = stats.compare(bench, ["control"], ["baseline"]) if controlled else None
    decision, why = stats.decide(cmp, noise=noise)
    reward = 100 * math.log1p(cmp["effect"]) if decision in ("KEEP", "SLOWER") else 0.
    if decision == "MIXED":
        reward = min(0., min(100 * math.log1p(s["effect"]) for s in cmp["shapes"].values() if s["ci"][1] < 0))
    return dict(record, decision=decision, reason=why, reward=reward, controlled=controlled,
                gain=1 + cmp["effect"], ci=[1 + v for v in cmp["ci"]], shapes=cmp["shapes"], noise=noise)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--validation", type=Path, help="Independent correctness stress receipt, optional")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if hashlib.sha256((ROOT / "release_2026-11-18/candidate.diff").read_bytes()).hexdigest() != BASE:
        raise ValueError("frozen release patch changed")
    spec = importlib.util.spec_from_file_location("kar_stats", PLUGIN)
    stats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats)
    pilot = score(args.pilot, stats)
    runs = [pilot]
    promoted = False
    if args.holdout:
        held = score(args.holdout, stats)
        for key in ("candidate", "scope", "agent", "candidate_sha256", "variant_identity"):
            if pilot[key] != held[key]:
                raise ValueError(f"holdout differs in {key}")
        if pilot["source_sha256"] == held["source_sha256"]:
            raise ValueError("holdout duplicates pilot")
        a, b = read(args.pilot), read(args.holdout)
        if a.get("seed") is None or b.get("seed") is None or a["seed"] == b["seed"]:
            raise ValueError("supply distinct input seeds for holdout")
        if not any(r["decision"] == "INCOMPLETE" for r in (pilot,held)):
            if set(a.get("bench", {}).get("shapes", {})) != set(b.get("bench", {}).get("shapes", {})):
                raise ValueError("holdout shape set differs")
            for k in a.get("bench", {}).get("shapes", {}):
                if a["bench"]["shapes"][k]["weight"] != b["bench"]["shapes"][k]["weight"]:
                    raise ValueError("holdout changed shape weights")
        runs.append(held)
        promoted = all(r["decision"] == "KEEP" and r["controlled"] for r in runs)
    output = dict(baseline_sha256=BASE, local_keep=promoted, runs=runs)
    if args.validation:
        validation = read(args.validation)
        if validation.get("baseline_sha256") != BASE or validation.get("candidate") != pilot["candidate"]:
            raise ValueError("stress receipt belongs to another candidate")
        if validation.get("correct") not in (True, False) or validation.get("active") is not True:
            raise ValueError("stress receipt needs active dispatch and explicit correctness")
        output["validation"] = dict(file=args.validation.resolve().relative_to(ROOT).as_posix(),
                                    sha256=hashlib.sha256(args.validation.read_bytes()).hexdigest(),
                                    correct=validation["correct"])
        if not validation["correct"]:
            output.update(local_keep=False, final_decision="INCORRECT", correctness_penalty=-100.0)
    if args.output:
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        tmp.replace(args.output)
    print(json.dumps({"candidate": pilot["candidate"], "local_keep": output["local_keep"],
                      "runs": [{k: r.get(k) for k in ("decision", "reward", "gain", "ci")} for r in runs]}))


if __name__ == "__main__":
    main()