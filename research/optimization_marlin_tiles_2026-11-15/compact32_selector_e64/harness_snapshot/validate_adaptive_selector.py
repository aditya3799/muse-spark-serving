"""Verify mutual exclusion and device-selected block cost through graph replay."""

import argparse
import json
import traceback
from collections import Counter
from pathlib import Path

from kernel_runtime import bootstrap
from tile_study import PATTERNS, make_routes, map_routes, manifest


def run(args, report):
    report["runtime"] = bootstrap(args.sglang_root)
    import torch
    from adaptive_tiles import prepare_adaptive_alignment

    if args.compact:
        from adaptive_tiles import (
            prepare_compact_alignment as prepare_adaptive_alignment,
        )
    from marlin_bench import capture
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

    report["checks"] = []
    width = args.small_block
    with torch.inference_mode():
        for tokens in (1, 8, 9, 32, 128, 256, 257, 1024):
            ids = torch.zeros((tokens, 8), device="cuda", dtype=torch.int32)
            for block in (32, 48, 64):
                if block <= width:
                    continue

                def call():
                    big = moe_align_block_size(ids, block, args.local_experts)
                    small = prepare_adaptive_alignment(
                        ids,
                        block,
                        args.local_experts,
                        big[2],
                        True,
                        big[0],
                        **({"small_block": width} if args.compact else {}),
                    )
                    return big[2], small[2], small[0], small[1]

                graph, output = capture(torch, call)
                for seed in (20261115, 20261116):
                    for pattern in PATTERNS:
                        mapped = map_routes(
                            make_routes(tokens, 8, args.global_experts, args.global_experts // args.local_experts, pattern, seed), args.local_experts, 0
                        )
                        counts = Counter(mapped)
                        active = [n for expert, n in counts.items() if expert >= 0]
                        small_blocks = sum((n + width - 1) // width for n in active)
                        big_blocks = sum((n + block - 1) // block for n in active)
                        choose_small = (
                            max(active, default=0) <= width
                            if args.compact
                            else small_blocks <= big_blocks
                        )
                        big_total = sum(
                            (n + block - 1) // block * block for n in counts.values()
                        )
                        small_total = sum(
                            (n + width - 1) // width * width
                            for n in (active if args.compact else counts.values())
                        )
                        ids.copy_(
                            torch.tensor(mapped, device="cuda", dtype=torch.int32).view(
                                tokens, 8
                            )
                        )
                        graph.replay()
                        got = [int(t.item()) for t in output[:2]]
                        expected = [0, small_total] if choose_small else [big_total, 0]
                        assert got == expected, (tokens, block, pattern, got, expected)
                        if args.compact and choose_small:
                            packed = output[2][:small_total].cpu().tolist()
                            experts = output[3][: small_total // width].cpu().tolist()
                            assert experts == sorted(e for e in counts if e >= 0)
                            seen = []
                            for b, expert in enumerate(experts):
                                for slot in packed[b * width : (b + 1) * width]:
                                    if slot != len(mapped):
                                        assert (
                                            0 <= slot < len(mapped)
                                            and mapped[slot] == expert
                                        )
                                        seen.append(slot)
                            assert sorted(seen) == [
                                i for i, expert in enumerate(mapped) if expert >= 0
                            ]
                        report["checks"].append(
                            dict(
                                tokens=tokens,
                                big_block=block,
                                seed=seed,
                                pattern=pattern,
                                chose_small=choose_small,
                                counts=got,
                            )
                        )
                del graph, output
    report.update(
        status="passed",
        cases=len(report["checks"]),
        compact=args.compact,
        small_block=width,
        local_experts=args.local_experts,
        global_experts=args.global_experts,
        scope="Actual CUDA alignment and Triton selection under changed-input graph replay; mutual exclusion; compact active-route conservation when enabled; no GEMM timing",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sglang-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--compact", action="store_true")
    p.add_argument("--small-block", type=int, choices=(16, 32), default=16)
    p.add_argument("--local-experts", type=int, default=32)
    p.add_argument("--global-experts", type=int, default=256)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"status": "starting"}
    try:
        run(args, report)
    except Exception:
        report.update(status="failed", traceback=traceback.format_exc())
    (args.out / "selector_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    manifest(args.out)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("runtime", "checks")},
            indent=2,
        )
    )
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
