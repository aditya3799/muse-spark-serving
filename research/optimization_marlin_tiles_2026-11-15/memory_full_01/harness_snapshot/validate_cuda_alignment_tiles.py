"""Execute the pinned generic CUDA aligner for all experimental tile sizes.

Uses the existing Windows NVRTC validation adapter. No alignment CUDA body is
edited, and no GEMM, Triton tiny aligner, or serving timing is substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from collections import Counter
from pathlib import Path

import torch

from tile_study import BLOCKS, PATTERNS, make_routes, map_routes, manifest


def verify(table, mapped, block):
    ordered, experts, count = table
    used = int(count.item())
    assert 0 <= used <= ordered.numel() and used % block == 0
    token_list = ordered[:used].cpu().tolist()
    expert_list = experts[: used // block].cpu().tolist()
    assert expert_list == sorted(expert_list)
    counts = Counter(mapped)
    expected_used = sum((n + block - 1) // block * block for n in counts.values())
    assert used == expected_used
    seen = []
    for index, expert in enumerate(expert_list):
        for slot in token_list[index * block : (index + 1) * block]:
            assert 0 <= slot <= len(mapped)
            if slot == len(mapped):
                continue
            assert mapped[slot] == expert
            seen.append(slot)
    assert sorted(seen) == list(range(len(mapped))), "Missing or repeated route slot"
    return used


def run(args, report):
    sys.path.insert(0, str(args.serving_root))
    import validate_alignment as align

    align.FUNCS = align.compile_kernels(align.preamble + align.body, align.NAMES)
    report.update(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda_source=str(align.AOT),
        cuda_source_sha256=hashlib.sha256(align.AOT.read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(align.__file__).read_bytes()).hexdigest(),
        scope="unmodified generic AOT alignment CUDA bodies via NVRTC; no tiny Triton aligner or GEMMs",
    )
    eager = 0
    for tokens in (1, 8, 9, 31, 32, 33, 127, 128, 129, 255, 256, 257, 513, 1024, 4096):
        for pattern in PATTERNS:
            raw = make_routes(tokens, 8, 256, 8, pattern, 20261115)
            for rank in (0, 1, 7):
                mapped = map_routes(raw, 32, rank)
                ids = torch.tensor(mapped, device="cuda", dtype=torch.int32).view(
                    tokens, 8
                )
                for block in BLOCKS:
                    verify(align.align(ids, 32, block), mapped, block)
                    eager += 1
        print(f"Eager routing complete: M={tokens}", flush=True)
    graph_checks = 0
    for tokens in (8, 9, 128, 513):
        for rank in (0, 7):
            ids = torch.zeros((tokens, 8), device="cuda", dtype=torch.int32)
            for block in BLOCKS:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        align.align(ids, 32, block)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    table = align.align(ids, 32, block)
                addresses = tuple(t.data_ptr() for t in table)
                for seed in range(2):
                    for pattern in PATTERNS:
                        raw = make_routes(tokens, 8, 256, 8, pattern, 61 + seed)
                        mapped = map_routes(raw, 32, rank)
                        ids.copy_(
                            torch.tensor(mapped, device="cuda", dtype=torch.int32).view(
                                tokens, 8
                            )
                        )
                        graph.replay()
                        assert tuple(t.data_ptr() for t in table) == addresses
                        verify(table, mapped, block)
                        graph_checks += 1
                del graph, table
    report.update(
        status="passed",
        eager_cases=eager,
        changed_input_graph_cases=graph_checks,
        full_moe_validated=False,
        performance_measured=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serving-root", type=Path, default=Path(r"D:\OpenKernelChallenge\serving")
    )
    parser.add_argument("--out", type=Path, required=True)
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
        (args.out / "alignment_validation.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        manifest(args.out)
        print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
