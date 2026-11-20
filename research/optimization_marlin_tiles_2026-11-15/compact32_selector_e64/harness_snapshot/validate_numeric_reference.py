"""Compare real Marlin expert paths against independent dequantized dense math."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from types import MethodType, SimpleNamespace

from kernel_runtime import bootstrap
from tile_study import PATTERNS, POLICIES, make_routes, manifest


def reference(torch, layer, x, scores, ids, norm_weight):
    local_ids = layer._expert_lookup[ids.to(torch.int32).clamp_min(-1) + 1]
    m, topk = ids.shape
    outputs = torch.zeros((m * topk, x.shape[-1]), device=x.device, dtype=x.dtype)
    for expert in range(layer.num_local_experts):
        slots = torch.nonzero(local_ids.flatten() == expert).flatten()
        if slots.numel() == 0:
            continue
        token_rows = slots // topk
        # FP32 dense accumulation then the stock BF16 stage boundaries.
        gate_up = (x[token_rows].float() @ layer.reference_w13[expert].float().T).to(
            x.dtype
        )
        gate, up = gate_up.float().chunk(2, dim=-1)
        activation = (torch.nn.functional.silu(gate) * up).to(x.dtype)
        outputs[slots] = (activation.float() @ layer.reference_w2[expert].float().T).to(
            x.dtype
        )
    values = outputs.view(m, topk, -1).float() * norm_weight.float()
    values *= torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    combined = torch.zeros((m, x.shape[-1]), device=x.device, dtype=torch.float32)
    for slot in range(topk):
        combined += values[:, slot] * scores[:, slot, None]
    return combined.to(x.dtype)


def metrics(torch, actual, expected):
    assert bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    error = actual.float() - expected.float()
    reference_norm = torch.linalg.vector_norm(expected.float()).item()
    relative_l2 = torch.linalg.vector_norm(error).item() / max(reference_norm, 1e-12)
    per_row = torch.linalg.vector_norm(error, dim=-1)
    row_scale = torch.linalg.vector_norm(expected.float(), dim=-1)
    denominator = row_scale.clamp_min(max(row_scale.mean().item() * 0.1, 1e-12))
    return {
        "relative_l2": relative_l2,
        "max_row_relative_l2": (per_row / denominator).max().item(),
        "max_abs": error.abs().max().item(),
    }


def run(args, report):
    report["runtime"] = bootstrap(args.sglang_root)
    import torch
    from marlin_bench import load_method, make_layer
    from sglang.kernels.ops.moe.muse_spark_post_expert_finalize_triton import (
        fused_post_expert_norm_weighted_sum,
    )

    if any(p.startswith("sorted_") for p in args.policies):
        from sorted_ep_experiment import install_dispatcher

        report["sorted_ep"] = install_dispatcher(
            args.sglang_root, args.out / "sorted_ep_overlay"
        )
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = SimpleNamespace(
        sglang_root=args.sglang_root,
        hidden=args.hidden,
        intermediate=args.intermediate,
        local_experts=args.local_experts,
        global_experts=args.global_experts,
        rank=0,
        retain_reference=True,
    )
    layer = make_layer(config, torch)
    norm_weight = torch.linspace(
        0.5, 1.5, args.hidden, device="cuda", dtype=torch.bfloat16
    )
    methods = {
        p: MethodType(load_method(args.sglang_root, p, torch)[0], layer)
        for p in args.policies
    }
    report.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__, rows=[])
    with torch.inference_mode():
        for m in args.tokens:
            for pattern in PATTERNS + ("duplicate_slots", "noncontiguous"):
                raw = make_routes(
                    m,
                    8,
                    args.global_experts,
                    args.global_experts // args.local_experts,
                    "uniform"
                    if pattern in ("duplicate_slots", "noncontiguous")
                    else pattern,
                    args.seed,
                )
                ids = torch.tensor(raw, device="cuda", dtype=torch.int32)
                if pattern == "duplicate_slots":
                    ids[:, :3] = 0
                    ids[:, -1] = -1
                if pattern == "noncontiguous":
                    ids = ids.repeat_interleave(2, dim=1)[:, ::2]
                    assert not ids.is_contiguous()
                x = (
                    torch.randn((m, args.hidden), device="cuda", dtype=torch.bfloat16)
                    / 20
                )
                scores = torch.rand((m, 8), device="cuda", dtype=torch.float32)
                scores.div_(scores.sum(-1, keepdim=True))
                expected = reference(torch, layer, x, scores, ids, norm_weight)
                for policy, method in methods.items():
                    values, local_ids = method(x, scores, ids)
                    actual = fused_post_expert_norm_weighted_sum(
                        values,
                        scores,
                        norm_weight=norm_weight,
                        eps=1e-6,
                        weight_first=True,
                        slot_expert_ids=local_ids,
                    )
                    row = dict(
                        tokens=m,
                        pattern=pattern,
                        policy=policy,
                        **metrics(torch, actual, expected),
                    )
                    report["rows"].append(row)
                    # Numerical screen, not a substitute for stock-token agreement.
                    assert row["relative_l2"] <= 0.015, row
                    assert row["max_row_relative_l2"] <= 0.075, row
                    if pattern in ("inactive", "all_remote"):
                        assert torch.count_nonzero(actual).item() == 0
            print(f"Independent reference checks complete: M={m}", flush=True)
    report.update(
        status="passed",
        cases=len(report["rows"]),
        max_relative_l2=max(r["relative_l2"] for r in report["rows"]),
        max_row_relative_l2=max(r["max_row_relative_l2"] for r in report["rows"]),
        scope="Synthetic NVFP4 weights; real kernels against independent dense reference; not serving validation",
        geometry={
            k: getattr(args, k)
            for k in ("hidden", "intermediate", "local_experts", "global_experts")
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--tokens", nargs="+", type=int, default=[1, 8, 9, 33, 129, 257]
    )
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--local-experts", type=int, default=32)
    parser.add_argument("--global-experts", type=int, default=256)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICIES
        + (
            "sorted_baseline",
            "sorted_ep_density",
            "adaptive_16",
            "adaptive_compact_16",
            "adaptive_compact_32",
        ),
        default=list(POLICIES),
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"status": "starting"}
    code = 0
    try:
        run(args, report)
    except Exception:
        code = 1
        report.update(status="failed", traceback=traceback.format_exc())
    finally:
        (args.out / "numeric_reference.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        manifest(args.out)
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("rows", "runtime")},
                indent=2,
            )
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
