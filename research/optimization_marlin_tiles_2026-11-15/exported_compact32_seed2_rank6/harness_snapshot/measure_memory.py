"""Measure one local-layer graph at a time; never claim full-model memory eligibility."""

import argparse
import gc
import json
import traceback
from pathlib import Path
from types import MethodType, SimpleNamespace

from kernel_runtime import bootstrap
from tile_study import make_routes, manifest


def run(args, report):
    report["runtime"] = bootstrap(args.sglang_root)
    import torch
    from marlin_bench import capture, load_method, make_layer
    from sglang.kernels.ops.moe.muse_spark_post_expert_finalize_triton import (
        fused_post_expert_norm_weighted_sum,
    )

    torch.manual_seed(20261115)
    layer = make_layer(
        SimpleNamespace(
            sglang_root=args.sglang_root,
            hidden=4096,
            intermediate=4096,
            local_experts=32,
            global_experts=256,
            rank=0,
        ),
        torch,
    )
    norm = torch.linspace(0.5, 1.5, 4096, device="cuda", dtype=torch.bfloat16)
    rows = report["rows"] = []
    with torch.inference_mode():
        for tokens in (128, 256, 1024):
            x = torch.randn((tokens, 4096), device="cuda", dtype=torch.bfloat16) / 20
            ids = torch.tensor(
                make_routes(tokens, 8, 256, 8, "uniform", 20261115),
                device="cuda",
                dtype=torch.int32,
            )
            scores = torch.rand((tokens, 8), device="cuda")
            scores /= scores.sum(-1, keepdim=True)
            for policy in ("baseline", "adaptive_compact_16", "adaptive_compact_32"):
                method = MethodType(
                    load_method(args.sglang_root, policy, torch)[0], layer
                )

                def call(method=method):
                    values, local = method(x, scores, ids)
                    return fused_post_expert_norm_weighted_sum(
                        values,
                        scores,
                        norm_weight=norm,
                        eps=1e-6,
                        weight_first=True,
                        slot_expert_ids=local,
                    )

                for _ in range(3):
                    call()
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                before = torch.cuda.memory_allocated()
                before_reserved = torch.cuda.memory_reserved()
                torch.cuda.reset_peak_memory_stats()
                graph, output = capture(torch, call)
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize()
                rows.append(
                    dict(
                        tokens=tokens,
                        policy=policy,
                        before_graph_bytes=before,
                        before_graph_reserved_bytes=before_reserved,
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        graph_peak_reserved_above_inputs_bytes=torch.cuda.max_memory_reserved()-before_reserved,
                        live_with_graph_reserved_bytes=torch.cuda.memory_reserved(),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        live_with_graph_bytes=torch.cuda.memory_allocated(),
                        graph_peak_above_inputs_bytes=torch.cuda.max_memory_allocated()
                        - before,
                    )
                )
                del graph, output, method, call
                gc.collect()
                torch.cuda.empty_cache()
    report.update(
        status="passed",
        scope="One local full-width layer graph at a time; excludes other model layers, communication, KV cache and graph-pool sharing; not the contest 2GB eligibility test",
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
    (args.out / "memory.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    manifest(args.out)
    print(json.dumps({k: v for k, v in report.items() if k != "runtime"}, indent=2))
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
