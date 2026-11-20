"""Separate instrumented trace of the complete synthetic local MoE path."""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

from kernel_runtime import bootstrap
from tile_study import PATTERNS, make_routes, manifest


def summarize_trace(path, out):
    trace = json.loads(path.read_text())
    events = trace["traceEvents"]
    ranges = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("name", "").startswith("stage:")
    ]
    launches = {
        e.get("args", {}).get("correlation"): e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver")
        and "Launch" in e.get("name", "")
    }
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    stats = defaultdict(lambda: {"kernels": 0, "kernel_us": 0.0})
    rows = []
    for event in kernels:
        launch = launches.get(event.get("args", {}).get("correlation"))
        stage = "unattributed"
        if launch:
            containing = [
                r
                for r in ranges
                if r.get("tid") == launch.get("tid")
                and r["ts"] <= launch["ts"] < r["ts"] + r["dur"]
            ]
            if containing:
                stage = min(containing, key=lambda r: r["dur"])["name"]
        stats[stage]["kernels"] += 1
        stats[stage]["kernel_us"] += event.get("dur", 0)
        rows.append(
            {
                "stage": stage,
                "name": event["name"],
                "duration_us": event.get("dur", 0),
                "stream": event.get("args", {}).get("stream"),
                "correlation": event.get("args", {}).get("correlation"),
            }
        )
    with (out / "kernels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["stage", "name", "duration_us", "stream", "correlation"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return {
        "kernel_count": len(kernels),
        "stages": dict(stats),
        "scope": "instrumented eager CUDA kernel durations; not graph latency or serving fractions",
        "unattributed_kernels": stats["unattributed"]["kernels"],
    }


def run(args, report):
    report["runtime"] = bootstrap(args.sglang_root)
    import torch
    import sglang.kernels.ops.activation.activation as activation
    import sglang.kernels.ops.moe.moe_align_small_numel as small_align
    import sglang.kernels.ops.moe.moe_wna16_marlin as marlin
    import sglang.srt.layers.moe.fused_moe_triton as generic_align
    from marlin_bench import load_method, make_layer
    from sglang.kernels.ops.moe.muse_spark_post_expert_finalize_triton import (
        fused_post_expert_norm_weighted_sum,
    )

    if args.policy.startswith("sorted_"):
        from sorted_ep_experiment import install_dispatcher

        report["sorted_ep"] = install_dispatcher(
            args.sglang_root, args.out / "sorted_ep_overlay"
        )
    torch.manual_seed(args.seed)
    config = SimpleNamespace(
        sglang_root=args.sglang_root,
        hidden=args.hidden,
        intermediate=args.intermediate,
        global_experts=256,
        local_experts=32,
        rank=args.rank,
    )
    layer = make_layer(config, torch)
    method = MethodType(load_method(args.sglang_root, args.policy, torch)[0], layer)
    x = (
        torch.randn((args.tokens, args.hidden), device="cuda", dtype=torch.bfloat16)
        / 20
    )
    ids = torch.tensor(
        make_routes(args.tokens, 8, 256, 8, args.pattern, args.seed),
        device="cuda",
        dtype=torch.int32,
    )
    scores = torch.rand((args.tokens, 8), device="cuda")
    scores.div_(scores.sum(-1, keepdim=True))
    norm = torch.linspace(0.5, 1.5, args.hidden, device="cuda", dtype=torch.bfloat16)

    def complete(annotate=False):
        values, local_ids = method(x, scores, ids)
        if annotate:
            with torch.profiler.record_function("stage:finalize"):
                return fused_post_expert_norm_weighted_sum(
                    values,
                    scores,
                    norm_weight=norm,
                    eps=1e-6,
                    weight_first=True,
                    slot_expert_ids=local_ids,
                )
        return fused_post_expert_norm_weighted_sum(
            values,
            scores,
            norm_weight=norm,
            eps=1e-6,
            weight_first=True,
            slot_expert_ids=local_ids,
        )

    def wrap(fn, name):
        def call(*a, **kw):
            label = name
            if name == "gemm":
                label = "gemm1" if kw["top_k"] == 8 else "gemm2"
            with torch.profiler.record_function("stage:" + label):
                return fn(*a, **kw)

        return call

    with torch.inference_mode():
        for _ in range(5):
            complete()
        torch.cuda.synchronize()
        with ExitStack() as stack:
            for module, name, label in (
                (marlin, "moe_wna16_marlin_gemm", "gemm"),
                (activation, "silu_and_mul", "activation"),
                (small_align, "moe_align_small_numel", "small_alignment"),
                (generic_align, "moe_align_block_size", "alignment"),
            ):
                stack.enter_context(
                    patch.object(module, name, wrap(getattr(module, name), label))
                )
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as prof:
                for _ in range(args.iterations):
                    with torch.profiler.record_function(
                        "stage:full_moe_including_mapping"
                    ):
                        complete(True)
                torch.cuda.synchronize()
            trace = args.out / "trace.json"
            prof.export_chrome_trace(str(trace))
    report.update(
        status="passed",
        gpu=torch.cuda.get_device_name(),
        summary=summarize_trace(trace, args.out),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=4096)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--policy", default="baseline")
    parser.add_argument("--pattern", choices=PATTERNS, default="uniform")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "starting",
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }
    code = 0
    try:
        run(args, report)
    except Exception:
        code = 1
        report.update(status="failed", traceback=traceback.format_exc())
    finally:
        (args.out / "profile.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        manifest(args.out)
        print(json.dumps({k: v for k, v in report.items() if k != "runtime"}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
