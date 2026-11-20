"""Collect already-downloaded cache summaries, log coverage, and capacity bounds."""
import json
import re
from pathlib import Path
import sys

here=Path(__file__).resolve().parent
root=here.parents[1]
sys.path.insert(0,str(root/"research_2026-11-18_expert_tp"))
from frontier_calibration import SOURCES

def summarize(source):
    record=json.loads((root/source).read_bytes())
    record=record.get("response",record)
    result=record.get("result") or {}
    raw=result.get("startup_log") or {}
    text=raw.get("text","")
    lines=text.splitlines()
    # Exclude server_args and setup chatter from operational counters.
    runtime=[l for l in lines if "server_args" not in l and "Namespace(" not in l and len(l)<1500]
    observed=[dict(new=int(a),cached=int(b)) for a,b in re.findall(r"#new-token:\s*(\d+),\s*#cached-token:\s*(\d+)",text)]
    operational = [l for l in runtime if re.search(r"evicted \d|prefetched \d|transferred \d|loaded \d.*(host|tier)|retracted \d|cached_tokens_details|cache_hit_rate_by_tier", l, re.I)]
    return dict(job=record["id"],source=source,status=record["status"],expected=result.get("expected_cache_hit"),
                rungs=[dict(users=r["concurrency"],reported_hit=r.get("cache_hit"),reps=r.get("reps"),
                            first_p90_s=r.get("ttft_first_p90_s"),later_p90_s=r.get("ttft_later_p90_s"))
                       for r in result.get("rungs",[])],
                stocks=[{"users":r["concurrency"],"reported_hit":r.get("stock",{}).get("cache_hit")}
                        for r in result.get("vs_stock",{}).get("rungs",[])],
                log_cut={k:v for k,v in raw.items() if k!="text"},retained_prefill_batches=observed,
                retained_runtime_tier_count_lines=operational)

paths=SOURCES+["research_2026-11-18_dflash_projection/mixed_chunk_judge.json",
               "research_2026-11-18_dflash_projection/projection_judge.json",
               "research_2026-11-18_target_moe/native_package_02/h200_job.json"]
runs=[summarize(p) for p in paths if (root/p).exists()]
inc=next(r for r in runs if r["job"]=="317c88a5fd474cd2")
official=json.loads((root/inc["source"]).read_bytes())["result"]
s=official["shape"]
turns=s["turns"]
new=s["first_turn"]+(turns-1)*s["new_per_turn"]
logical=turns*s["first_turn"]+turns*(turns-1)//2*(s["new_per_turn"]+s["output_mean"])
state=official["launch_settings"]["derived"]
capacity=next(row["build"] for row in state if row["setting"]=="internal_states[0].memory_usage.token_capacity")
physical=sorted(set(re.findall(r"Allocating (.{1,50}?) host pool: ([\d,]+) tokens",official["startup_log"]["text"])))
output=dict(runs=runs, standalone_previous_logs={}, official={"shape":s,"nominal_logical_input_rows":logical,
            "nominal_unseen_input_rows":new,"nominal_reuse_ceiling_private_independent_conversations":1-new/logical,
            "ceiling_assumptions":"13 turns, fixed mean new rows and outputs, identical retained own history, zero prefix shared between unrelated conversations; not a hard bound for undisclosed trace content",
            "published_expected_hit":inc["expected"],"hbm_full_token_capacity":capacity,"host_pools_in_startup":physical,
            "nominal_32_full_first_context_rows_over_hbm_capacity":32*s["first_turn"]-capacity,
            "nominal_16_full_first_context_rows_over_hbm_capacity":16*s["first_turn"]-capacity,
            "capacity_assumptions":"All 32 private contexts concurrently live and fully materialized, no sharing; actual occupancy/evictions unreported",
            "reported_hit_rounding_conditional":{"if_nearest_two_decimal_lower_bound":0.915,
            "max_avoidable_miss_share_of_logical_rows_with_nominal_ceiling":1-new/logical-0.915,
            "max_share_of_logical_rows_missed_any_cause":0.085,"rounding_rule_disclosed":False},
            "release_score":official["frontier"]["score"]})
for p in root.glob("startup*log"):
    text=p.read_text(encoding="utf-8",errors="replace")
    matches=re.findall(r"#new-token:\s*(\d+),\s*#cached-token:\s*(\d+)",text)
    output["standalone_previous_logs"][p.name]=dict(chars=len(text),prefill_batches=matches)
(here/"cache_receipts.json").write_text(json.dumps(output,indent=2))
print("summaries:",[(r["job"][:8],[p["reported_hit"] for p in r["rungs"]]) for r in runs])
print("incumbent:",inc["log_cut"],inc["retained_prefill_batches"],inc["retained_runtime_tier_count_lines"],physical)
print("nominal logical, new, reuse:",logical,new,1-new/logical)
