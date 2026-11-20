"""Serial virtual-rank timing of the exported patch, including the slowest shard."""

import argparse
import json
import statistics
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import marlin_bench
from tile_study import manifest
from validate_exported_candidate import main as exported_main


def rank_main():
    p = argparse.ArgumentParser()
    p.add_argument("--sglang-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ranks", nargs="+", type=int, default=list(range(8)))
    p.add_argument("--tokens", nargs="+", type=int, default=[256, 512])
    p.add_argument("--samples", type=int, default=6)
    p.add_argument("--repetitions", type=int, default=6)
    args = p.parse_args()
    import torch

    original_make_layer = marlin_bench.make_layer
    shared = None

    def make_shared(config, torch_module):
        nonlocal shared
        if shared is None:
            shared = original_make_layer(config, torch_module)
        else:
            shared._expert_map.fill_(-1)
            start = config.rank * config.local_experts
            shared._expert_map[start : start + config.local_experts] = torch.arange(
                config.local_experts, device="cuda", dtype=torch.int32
            )
            shared._expert_lookup[1:].copy_(shared._expert_map)
        return shared

    marlin_bench.make_layer = make_shared
    all_reports = []
    for rank in args.ranks:
        out = args.out / f"rank_{rank}"
        out.mkdir()
        config = SimpleNamespace(
            sglang_root=args.sglang_root,
            out=out,
            mode="bench",
            hidden=4096,
            intermediate=4096,
            global_experts=256,
            local_experts=32,
            rank=rank,
            topk=8,
            tokens=args.tokens,
            policies=["baseline", "adaptive_compact_32"],
            patterns=[
                "uniform",
                "hot_rank",
                "hot_expert",
                "repeated",
                "all_remote",
                "inactive",
            ],
            seed=20261115,
            samples=args.samples,
            repetitions=args.repetitions,
            rtol=0.01,
            atol=0.01,
            replay_atol=0.0078125,
            replay_relative_l2=0.0005,
        )
        report = dict(
            status="starting",
            started_unix=time.time(),
            arguments={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(config).items()
            },
            virtual_rank=rank,
            local_weight_values_reused=True,
        )
        try:
            report.update(marlin_bench.preflight(args.sglang_root))
            with torch.inference_mode():
                marlin_bench.run_benchmark(config, report)
        except Exception:
            report.update(status="failed", traceback=traceback.format_exc())
        report["finished_unix"] = time.time()
        (out / "benchmark.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        manifest(out)
        if report["status"] != "passed":
            print(report.get("traceback", report["status"]), flush=True)
            return 1
        all_reports.append(report)
        print(f"Completed virtual rank {rank}", flush=True)
    rows = []
    for tokens in args.tokens:
        for pattern in config.patterns:
            rank_rows = []
            for report in all_reports:
                row = next(
                    r
                    for r in report["timing_rows"]
                    if r["tokens"] == tokens
                    and r["pattern"] == pattern
                    and r["schedule"] == "graph"
                    and r["policy"] == "adaptive_compact_32"
                )
                rank_rows.append(
                    dict(
                        rank=report["virtual_rank"],
                        baseline_us=statistics.median(
                            s["baseline"] for s in row["samples_us"]
                        ),
                        candidate_us=statistics.median(
                            s["candidate"] for s in row["samples_us"]
                        ),
                    )
                )
            baseline = max(r["baseline_us"] for r in rank_rows)
            candidate = max(r["candidate_us"] for r in rank_rows)
            rows.append(
                dict(
                    tokens=tokens,
                    pattern=pattern,
                    max_baseline_us=baseline,
                    max_candidate_us=candidate,
                    serial_max_rank_ratio=baseline / candidate,
                    ranks=rank_rows,
                )
            )
    result = dict(
        status="passed",
        ranks=args.ranks,
        scope="Sequential virtual ranks on one RTX3050 with shared synthetic local weight values; max-rank timing proxy only, no distributed overlap/all-reduce or serving proof",
        rows=rows,
    )
    (args.out / "serial_rank_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    manifest(args.out)
    print(
        json.dumps(
            {
                **result,
                "rows": [{k: v for k, v in r.items() if k != "ranks"} for r in rows],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    marlin_bench.main = rank_main
    raise SystemExit(exported_main())
