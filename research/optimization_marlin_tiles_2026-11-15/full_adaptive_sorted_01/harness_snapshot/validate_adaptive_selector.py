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
    from marlin_bench import capture
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    report["checks"] = []
    with torch.inference_mode():
        for tokens in (1, 8, 9, 32, 128, 256, 257, 1024):
            ids = torch.zeros((tokens, 8), device="cuda", dtype=torch.int32)
            for block in (32, 48, 64):
                def call():
                    big = moe_align_block_size(ids, block, 32)
                    small = prepare_adaptive_alignment(ids, block, 32, big[2], True)
                    return big[2], small[2]
                graph, output = capture(torch, call)
                for seed in (20261115, 20261116):
                    for pattern in PATTERNS:
                        mapped = map_routes(make_routes(tokens, 8, 256, 8, pattern, seed), 32, 0)
                        counts = Counter(mapped)
                        active = [n for expert, n in counts.items() if expert >= 0]
                        small_blocks = sum((n + 15) // 16 for n in active)
                        big_blocks = sum((n + block - 1) // block for n in active)
                        choose_small = small_blocks <= big_blocks
                        big_total = sum((n + block - 1) // block * block for n in counts.values())
                        small_total = sum((n + 15) // 16 * 16 for n in counts.values())
                        ids.copy_(torch.tensor(mapped, device="cuda", dtype=torch.int32).view(tokens, 8))
                        graph.replay()
                        got = [int(t.item()) for t in output]
                        expected = [0, small_total] if choose_small else [big_total, 0]
                        assert got == expected, (tokens, block, pattern, got, expected)
                        report["checks"].append(dict(tokens=tokens, big_block=block, seed=seed, pattern=pattern, chose_small=choose_small, counts=got))
                del graph, output
    report.update(status="passed", cases=len(report["checks"]), scope="Actual CUDA aligners and Triton selection under changed-input graph replay; exactly one nonzero routing table; no GEMM timing")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sglang-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"status": "starting"}
    try:
        run(args, report)
    except Exception:
        report.update(status="failed", traceback=traceback.format_exc())
    (args.out / "selector_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest(args.out)
    print(json.dumps({k:v for k,v in report.items() if k not in ("runtime", "checks")}, indent=2))
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
