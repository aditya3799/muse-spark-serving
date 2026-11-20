"""Summarize paired measurements without pooling inactive work into speedups."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

from tile_study import make_routes, map_routes


def percentile(values, q):
    values = sorted(values)
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - pos) + values[hi] * (pos - lo) if hi != lo else values[lo]


def summarize(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    args = report["arguments"]
    rng = random.Random(901)
    rows = []
    for row in report.get("timing_rows", []):
        ratios = [s["baseline"] / s["candidate"] for s in row["samples_us"]]
        draws = [
            statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000)
        ]
        routes = make_routes(
            row["tokens"],
            args["topk"],
            args["global_experts"],
            args["global_experts"] // args["local_experts"],
            row["pattern"],
            args["seed"],
        )
        mapped = map_routes(routes, args["local_experts"], args["rank"])
        counts = Counter(v for v in mapped if v >= 0)
        item = {
            k: row[k]
            for k in (
                "tokens",
                "policy",
                "pattern",
                "schedule",
                "baseline_block",
                "candidate_block",
            )
        }
        item.update(
            active_routes=sum(counts.values()),
            active_experts=len(counts),
            baseline_us=statistics.median(s["baseline"] for s in row["samples_us"]),
            candidate_us=statistics.median(s["candidate"] for s in row["samples_us"]),
            speedup=statistics.median(ratios),
            paired_bootstrap_low=percentile(draws, 0.025),
            paired_bootstrap_high=percentile(draws, 0.975),
        )
        if row["policy"].startswith("adaptive_"):
            big = row["baseline_block"]
            width = 32 if row["policy"].endswith("_32") else 16
            small_blocks = sum((n + width - 1) // width for n in counts.values())
            big_blocks = sum((n + big - 1) // big for n in counts.values())
            if (
                big > width
                and len(mapped) <= 8192
                and args["local_experts"]
                <= (128 if row["policy"].startswith("adaptive_compact_") else 1024)
                and (
                    max(counts.values(), default=0) <= width
                    if row["policy"].startswith("adaptive_compact_")
                    else small_blocks <= big_blocks
                )
            ):
                item["candidate_block"] = width
        for variant in ("baseline", "candidate"):
            block = item[variant + "_block"]
            item[variant + "_active_blocks"] = sum(
                (n + block - 1) // block for n in counts.values()
            )
        rows.append(item)
    groups = {}
    for schedule in ("graph", "eager"):
        for policy in args["policies"]:
            selected = [
                r
                for r in rows
                if r["schedule"] == schedule
                and r["policy"] == policy
                and r["active_routes"] > 0
            ]
            if selected:
                groups[schedule + ":" + policy] = {
                    "rows": len(selected),
                    "geomean_component_ratio": math.exp(
                        statistics.mean(math.log(r["speedup"]) for r in selected)
                    ),
                    "minimum_ratio": min(r["speedup"] for r in selected),
                    "maximum_ratio": max(r["speedup"] for r in selected),
                    "regressions_below_0_95": sum(
                        r["speedup"] < 0.95 for r in selected
                    ),
                }
    output = {
        "status": report["status"],
        "source": str(path),
        "scope": "Synthetic single-shard component timings; ratios are relative to V4 cached-mapping baseline, not stock serving. Bootstrap intervals reflect sampled pairs only, not device/system uncertainty.",
        "checks": len(report.get("checks", [])),
        "max_abs": max((c["max_abs"] for c in report.get("checks", [])), default=None),
        "bitwise_checks": sum(c["bitwise_equal"] for c in report.get("checks", [])),
        "groups": groups,
        "rows": rows,
    }
    (path.parent / "analysis.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    if rows:
        with (path.parent / "analysis.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({k: v for k, v in output.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    summarize(parser.parse_args().benchmark)
