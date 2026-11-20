"""Measure unchanged-baseline repeatability before accepting replay tolerances."""

import argparse
import json
import traceback
from pathlib import Path
from types import MethodType, SimpleNamespace

from kernel_runtime import bootstrap
from tile_study import make_routes, manifest


def metrics(torch, a, b):
    delta = a.float() - b.float()
    return dict(
        bitwise_equal=torch.equal(a, b),
        mismatched=int((a != b).sum()),
        max_abs=float(delta.abs().max()),
        relative_l2=float(delta.norm() / b.float().norm().clamp_min(1e-20)),
        finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
    )


def run(args, report):
    report["runtime"] = bootstrap(args.sglang_root)
    import torch
    from marlin_bench import capture, load_method, make_layer
    from sglang.kernels.ops.moe.muse_spark_post_expert_finalize_triton import (
        fused_post_expert_norm_weighted_sum,
    )

    torch.manual_seed(20261115)
    config = SimpleNamespace(
        sglang_root=args.sglang_root,
        hidden=4096,
        intermediate=4096,
        global_experts=256,
        local_experts=32,
        rank=0,
    )
    layer = make_layer(config, torch)
    method = MethodType(load_method(args.sglang_root, "baseline", torch)[0], layer)
    norm = torch.linspace(0.5, 1.5, 4096, device="cuda", dtype=torch.bfloat16)
    rows = report["rows"] = []
    with torch.inference_mode():
        for tokens in (128, 256):
            x = torch.randn((tokens, 4096), device="cuda", dtype=torch.bfloat16) / 20
            scores = torch.rand((tokens, 8), device="cuda")
            scores /= scores.sum(-1, keepdim=True)
            ids = torch.tensor(
                make_routes(tokens, 8, 256, 8, "uniform", 20261115),
                device="cuda",
                dtype=torch.int32,
            )

            def call():
                values, local = method(x, scores, ids)
                return fused_post_expert_norm_weighted_sum(
                    values,
                    scores,
                    norm_weight=norm,
                    eps=1e-6,
                    weight_first=True,
                    slot_expert_ids=local,
                )

            graph, graph_out = capture(torch, call)
            for pattern in ("uniform", "hot_rank", "hot_expert", "repeated"):
                ids.copy_(
                    torch.tensor(
                        make_routes(tokens, 8, 256, 8, pattern, 20261115),
                        device="cuda",
                        dtype=torch.int32,
                    )
                )
                eager_ref = call().clone()
                graph.replay()
                graph_ref = graph_out.clone()
                for i in range(12):
                    eager = call()
                    graph.replay()
                    for comparison, a, b in (
                        ("eager_vs_eager", eager, eager_ref),
                        ("graph_vs_graph", graph_out, graph_ref),
                        ("graph_vs_eager", graph_out, eager),
                    ):
                        rows.append(
                            dict(
                                tokens=tokens,
                                pattern=pattern,
                                iteration=i,
                                comparison=comparison,
                                **metrics(torch, a, b),
                            )
                        )
                print(f"Diagnosed M={tokens} {pattern}", flush=True)
            del graph, graph_out
    report.update(
        status="passed",
        scope="Unmodified V4 local MoE baseline; synthetic full-dimension BF16 NVFP4 path, no policy changes",
        max_abs=max(r["max_abs"] for r in rows),
        max_relative_l2=max(r["relative_l2"] for r in rows),
        non_bitwise_comparisons=sum(not r["bitwise_equal"] for r in rows),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sglang-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"status": "starting"}
    try:
        run(args, report)
    except Exception:
        report.update(status="failed", traceback=traceback.format_exc())
    (args.out / "repeatability.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    manifest(args.out)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("runtime", "rows")}, indent=2
        )
    )
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
