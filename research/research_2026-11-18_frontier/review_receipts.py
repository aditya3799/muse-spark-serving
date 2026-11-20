"""Read the three already queued serving receipts and save comparable signals locally."""
import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = HERE.parent
JOBS = {
    "mixed_chunk": ("b2fa80590214468b",S/"research_2026-11-18_dflash_projection/mixed_chunk_judge.json"),
    "dflash_projection": ("76866640ab6f4b9f",S/"research_2026-11-18_dflash_projection/projection_judge.json"),
    "hopper_experts": ("0afdc0b191dd4b42",S/"research_2026-11-18_target_moe/native_package_02/h200_job.json"),
}
FINISHED = {"done","failed","canceled","cancelled","rejected","timed_out","error","passed","ok","succeeded","completed"}


def inspect(receipt, job, calibration):
    if receipt.get("id") != job or receipt.get("mode")!="check":
        raise ValueError("unexpected serving receipt")
    state = receipt.get("status")
    result = receipt.get("result") or {}
    data=dict(job=job, state=state, content_hash=receipt.get("content_hash"),
              finished=bool(receipt.get("finished_at")) or state in FINISHED)
    if not result:
        return data
    reports = result.get("vs_stock",{}).get("rungs",[])
    captured = {r["concurrency"]:r for r in result.get("rungs",[])}
    checks=[]
    for row in reports:
        users=row["concurrency"]
        candidate=captured.get(users,{})
        stock=row.get("stock",{})
        raw = (candidate["tok_s_per_gpu"]/stock["tok_s_per_gpu"]
               if candidate.get("tok_s_per_gpu") and stock.get("tok_s_per_gpu") else None)
        checks.append(dict(users=users, raw_gpu_throughput_ratio=raw, samples=candidate.get("reps"),
                           first_ttft_ratio=row.get("ttft_first_p90"),later_ttft_ratio=row.get("ttft_later_p90"),
                           acceptance=candidate.get("acceptance"),
                           extrapolated_historical_low_factor_times_raw=(calibration*raw if users==16 and raw else None)))
    data.update(judge_pass=result.get("passed"),correctness=result.get("correctness",{}).get("passed"),
                later_turns=result.get("later_turns",{}).get("passed"),short_outputs=result.get("short_outputs"),
                memory=result.get("gpu_memory"),loads=checks)
    return data


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--watch",action="store_true")
    args=parser.parse_args()
    calibration=json.loads((S/"research_2026-11-18_expert_tp/frontier_calibration.json").read_text())
    factor=min(r["score_over_raw_c16"] for r in calibration["scored"])
    target=HERE/"judge_review.json"
    while True:
        jobs={}
        for label,(job,path) in JOBS.items():
            try:
                jobs[label]=inspect(json.loads(path.read_text(encoding="utf-8-sig")),job,factor)
            except (OSError,ValueError) as error:
                jobs[label]=dict(job=job,finished=False,read_error=str(error))
        report=dict(jobs=jobs,minimum_requested_frontier=1.8,
                    interpretation="A one-load check may justify closer review. No existing one-load result or historical multiplier establishes a four-load score. Approval is required for any scored submission.")
        content=json.dumps(report,indent=2)+"\n"
        if not target.exists() or target.read_text()!=content:
            temp=target.with_suffix(".tmp")
            temp.write_text(content)
            temp.replace(target)
        if not args.watch or all(r["finished"] for r in jobs.values()):
            return
        time.sleep(60)


if __name__=="__main__":
    main()

