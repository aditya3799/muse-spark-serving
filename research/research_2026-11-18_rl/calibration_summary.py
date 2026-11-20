"""Summarize the exact stock-versus-scored local calibration without mixing baselines."""
import json
import statistics
from pathlib import Path

here=Path(__file__).resolve().parent
source=here.parent/"research_2026-11-18_rl_prefill/calibration"
result=dict(scope="single RTX 3050 ranked expert path, pinned stock Marlin vs release Humming",runs={},excluded={})
extra=sorted(path.name for path in source.glob("wide_retry_*") if path.is_dir())
for split in ["pilot","holdout","wide_pilot","wide_holdout",*extra]:
    path=source/split/"calibration.json"
    if not path.exists():
        continue
    data=json.loads(path.read_text())
    if not data["complete"]:
        result["excluded"][split]="Interrupted before all duplicate controls and shape measurements completed"
    assert data["shapes"]
    rows={}
    for name,s in data["shapes"].items():
        if len(s["numerics"]) != data["rounds"] and not s.get("failed_controls"):
            result["excluded"][split + "/" + name] = "Incomplete duplicate control and numerical series"
            continue
        if s.get("failed_controls"):
            result["excluded"][split + "/" + name] = s["failed_controls"]
            continue
        t=s["times_us"]
        med=statistics.median
        rows[name]=dict(stock_ms=med(t["stock"])/1000,scored_parent_ms=med(t["scored"])/1000,
            median_paired_speedup=med(a/b for a,b in zip(t["stock"],t["scored"])),
            stock_control_ratio=med(a/b for a,b in zip(t["stock"],t["stock_copy"])),
            max_relative_l2=max(r["relative_l2"] for r in s["numerics"]),rounds=len(t["stock"]))
    result["runs"][split]=dict(completed_run=data["complete"],source=path.relative_to(here.parent).as_posix(),shapes=rows)
(here/"calibration_summary.json").write_text(json.dumps(result,indent=2)+"\n")