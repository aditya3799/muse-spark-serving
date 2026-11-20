"""Historical 16-user pseudo-check calibration. Read only saved OKC judge receipts."""
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCES = [
    "result_2dbd8ecf4c7c494b.json",
    "result_5183861a98ee4825.json",
    "research_iteration_2026-11-15/scored_468701af35b84e07.json",
    "swarm_2026-11-16/coordination/status_e399c79bb1994630.json",
    "release_2026-11-18/scored_submission_status.json",
    "result_717001a114f3471a.json",
    "swarm_2026-11-16/coordination/status_f1513498e8d84de8.json",
    "swarm_2026-11-16/coordination/recovery_20261116/status_9689b3114c2a45ab.json",
    "release_2026-11-18/evidence/status_fa0d2a0fcf584339.json",
    "release_2026-11-18/evidence/status_6570fbd9e1b64ce4.json",
]

def bracket_error(rungs, point):
    target = point["tok_s_per_user"]
    for left, right in zip(rungs, rungs[1:]):
        xl, xr = left["tok_s_per_user_e2e"], right["tok_s_per_user_e2e"]
        if xl >= target >= xr:
            yl, yr = left["tok_s_per_gpu"], right["tok_s_per_gpu"]
            linear = yl + (target-xl)/(xr-xl)*(yr-yl)
            alpha = (math.log(target)-math.log(xl))/(math.log(xr)-math.log(xl))
            loglog = math.exp(math.log(yl) + alpha*(math.log(yr)-math.log(yl)))
            return {"linear": linear / point["tok_s_per_gpu"] - 1,
                    "log_log": loglog / point["tok_s_per_gpu"] - 1}
    return None

def refs():
    scored, checks = [], []
    for source in SOURCES:
        receipt = json.loads((ROOT / source).read_bytes())
        receipt = receipt.get("response", receipt)
        r = receipt["result"]
        assert r["status"] == "ok" and r["passed"]
        info = dict(job=receipt["id"], patch=receipt["content_hash"], path=source,
                    created_utc_epoch=receipt["created_at"])
        if receipt["mode"] == "check":
            raw = r["rungs"][0]
            vs = r["vs_stock"]["rungs"][0]
            ratio = raw["tok_s_per_gpu"] / vs["stock"]["tok_s_per_gpu"]
            assert abs(ratio - vs["at_stock_point"]) < .00051
            checks.append(dict(**info, users=raw["concurrency"], repetitions=raw["reps"],
                candidate_gpu_tok_s=raw["tok_s_per_gpu"],stock_gpu_tok_s=vs["stock"]["tok_s_per_gpu"],
                raw_ratio=ratio, reported_at_stock_point=vs["at_stock_point"],
                separate_comparison_tok_s_per_gpu=vs["tok_s_per_gpu"]))
            continue
        f = r["frontier"]
        per_load=[]
        for rung, pt in zip(r["rungs"], f["points"]):
            assert rung["concurrency"] == pt["concurrency"]
            quotient=pt["tok_s_per_gpu"]/pt["stock_tok_s_per_gpu"]
            assert abs(quotient-pt["ratio"]) < .00006
            per_load.append(dict(users=rung["concurrency"], raw_gpu=rung["tok_s_per_gpu"],
                raw_user_e2e=rung["tok_s_per_user_e2e"], reps=rung["reps"], spread=rung["spread"],
                frontier_candidate_gpu=pt["tok_s_per_gpu"], frontier_stock_gpu=pt["stock_tok_s_per_gpu"],
                frontier_stock_user=pt["tok_s_per_user"], frontier_ratio=pt["ratio"],
                interpolation_relative_error=bracket_error(r["rungs"],pt)))
        assert [p["users"] for p in per_load] == [1,4,16,32]
        reconstructed=statistics.geometric_mean(p["frontier_ratio"] for p in per_load)
        assert abs(reconstructed-f["score"]) < .00011
        c16 = per_load[2]
        x = c16["raw_gpu"] / c16["frontier_stock_gpu"]
        scored.append(dict(**info, score=f["score"], paired_reps=f["paired_reps"],
            raw_c16_ratio=x, score_over_raw_c16=f["score"]/x,
            reconstructed_geomean=reconstructed, per_load=per_load))
    return sorted(scored,key=lambda q:q["created_utc_epoch"]), sorted(checks,key=lambda q:q["created_utc_epoch"])

def main():
    scored,checks=refs()
    scores_by_hash={r["patch"] for r in scored}
    assert len(scores_by_hash)==len(scored)==5
    same_patch=[c["job"] for c in checks if c["patch"] in scores_by_hash]
    assert not same_patch
    held_out=[]
    for r in scored:
        train=[p["score_over_raw_c16"] for p in scored if p["job"]!=r["job"]]
        pred=r["raw_c16_ratio"]*statistics.median(train)
        held_out.append(dict(job=r["job"],actual=r["score"],prediction=pred,
            relative_error=pred/r["score"]-1,raw_unadjusted_error=r["raw_c16_ratio"]/r["score"]-1))
    prev,latest=scored[-2:]
    pred=latest["raw_c16_ratio"]*prev["score_over_raw_c16"]
    interpolation=[p["interpolation_relative_error"] for r in scored for p in r["per_load"]
                   if p["interpolation_relative_error"] is not None]
    ranges=[r["score_over_raw_c16"] for r in scored]
    result=dict(
       method="The unscored metric's rounded at_stock_point equals raw candidate/stock at the same concurrency in all five saved checks. For each independent scored job, a pseudo-check uses its aggregated candidate C16 raw GPU tok/s / its paired C16 stock frontier GPU tok/s. Unlike actual C16 checks (one rep), scored C16 summaries report three reps. Each holdout scales this observable by the median score-to-pseudo-check ratio across the other four jobs. A retrospective diagnostic, not an independent digital twin of a one-rep check.",
       measurement_limits="Receipts return rung and paired-repetition counts plus aggregate values and spread. No individual candidate or stock repeat sequence. No saved C16 check shares a scored patch hash. No test was run near 1.8; kernel/layout changes can shift the load curve substantially. No calibration uncertainty interval follows from these five related patches.",
       exact_hash_pairs=same_patch, scored=scored,checks=checks,held_out=held_out,
       historical_relative_error_mean_absolute=statistics.mean(abs(r["relative_error"]) for r in held_out),
       historical_relative_error_worst_absolute=max(abs(r["relative_error"]) for r in held_out),
       prior_to_incumbent=dict(training_job=prev["job"],heldout_job=latest["job"],estimate=pred,
           actual=latest["score"],relative_error=pred/latest["score"]-1),
       empirical_multiplier_range=[min(ranges),max(ranges)],
       raw_c16_to_yield_1p8_using_smallest_observed_factor=1.8/min(ranges),
       historical_max_raw_c16=max(r["raw_c16_ratio"] for r in scored),
       observed_interpolation=dict(bracketed_points=len(interpolation),total_points=20,
           mean_absolute_relative_error={k:statistics.mean(abs(p[k]) for p in interpolation) for k in ("linear","log_log")},
           worst_absolute_relative_error={k:max(abs(p[k]) for p in interpolation) for k in ("linear","log_log")},
           comment="Tested linear and log-log interpolation between published aggregate E2E rung speeds and GPU throughputs. Printed points are rounded. Neither reconstructed all official points; raw per-repetition inputs, policy at endpoints, and judge implementation absent."))
    (HERE/"frontier_calibration.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("scored","checks","method","measurement_limits")}))
if __name__ == "__main__":
    main()

