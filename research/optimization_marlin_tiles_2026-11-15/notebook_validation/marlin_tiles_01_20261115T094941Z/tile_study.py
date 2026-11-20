"""EP tile geometry study. Counts are structural proxies, never GPU timings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

BLOCKS = (8, 16, 32, 48, 64)
PATTERNS = ("uniform", "hot_rank", "hot_expert", "repeated", "all_remote", "inactive")
TOKEN_COUNTS = (1, 7, 8, 9, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1024, 4096)
POLICIES = ("baseline", "ep_density") + tuple(f"fixed_{b}" for b in BLOCKS)


def choose_block(tokens, topk, local_experts, global_experts, policy):
    if tokens < 0 or not 0 < local_experts <= global_experts or topk <= 0:
        raise ValueError("Invalid geometry")
    if policy.startswith("fixed_"):
        block = int(policy[6:])
        if block not in BLOCKS:
            raise ValueError("Unsupported block")
        return block
    if policy not in ("baseline", "ep_density"):
        raise ValueError("Unknown policy")
    denominator = local_experts if policy == "baseline" else global_experts
    for block in BLOCKS:
        if tokens * topk / denominator / block < 0.9:
            break
    return block


def make_routes(tokens, topk, experts, ep, pattern, seed):
    if experts % ep or experts < topk:
        raise ValueError("Experts must divide ranks and cover top-k")
    rng = random.Random(seed)
    local = experts // ep
    if pattern not in PATTERNS:
        raise ValueError(pattern)
    repeated = rng.sample(range(experts), topk)
    rows = []
    for _ in range(tokens):
        if pattern == "inactive":
            row = [-1] * topk
        elif pattern == "all_remote":
            # Remote to rank zero; these are still real work on other ranks.
            pool = list(range(local, experts)) if ep > 1 else []
            row = rng.sample(pool, topk) if len(pool) >= topk else [-1] * topk
        elif pattern == "repeated":
            row = repeated[:]
        elif pattern == "hot_rank":
            pool = list(range(local))
            row = (
                rng.sample(pool, topk)
                if local >= topk
                else rng.sample(range(experts), topk)
            )
        elif pattern == "hot_expert":
            row = [0] + rng.sample(range(1, experts), topk - 1)
        else:
            row = rng.sample(range(experts), topk)
        rows.append(row)
    return rows


def map_routes(rows, local, rank):
    start = local * rank
    return [
        x - start if start <= x < start + local else -1 for row in rows for x in row
    ]


def pack_reference(mapped, local, block):
    """CPU specification only; not stock alignment implementation or ordering proof."""
    buckets = [[] for _ in range(local + 1)]
    for slot, expert in enumerate(mapped):
        if not -1 <= expert < local:
            raise ValueError("Invalid local expert")
        buckets[expert + 1].append(slot)
    token_ids, expert_ids = [], []
    sentinel = len(mapped)
    for bucket, slots in enumerate(buckets):
        padded = ((len(slots) + block - 1) // block) * block
        token_ids.extend(slots + [sentinel] * (padded - len(slots)))
        expert_ids.extend([bucket - 1] * (padded // block))
    return token_ids, expert_ids


def verify_pack(mapped, token_ids, expert_ids, block):
    if len(token_ids) != block * len(expert_ids):
        raise AssertionError("Malformed block table")
    seen = []
    for i, expert in enumerate(expert_ids):
        for slot in token_ids[i * block : (i + 1) * block]:
            if slot == len(mapped):
                continue
            if not 0 <= slot < len(mapped) or mapped[slot] != expert:
                raise AssertionError("Route moved to wrong expert")
            seen.append(slot)
    if sorted(seen) != list(range(len(mapped))):
        raise AssertionError("Lost or duplicate route")


def structure_study(out, seeds=4):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records, verified = [], 0
    for ep in (1, 2, 8):
        local, global_experts, topk = 256 // ep, 256, 8
        for tokens in TOKEN_COUNTS:
            for pattern in PATTERNS:
                for seed in range(seeds):
                    routes = make_routes(
                        tokens, topk, global_experts, ep, pattern, seed
                    )
                    ranks = [map_routes(routes, local, rank) for rank in range(ep)]
                    # Every nonnegative global route belongs to exactly one rank.
                    assert sum(sum(x >= 0 for x in rank) for rank in ranks) == sum(
                        x >= 0 for row in routes for x in row
                    )
                    for block in BLOCKS:
                        for rank, mapped in enumerate(ranks):
                            counts = Counter(x for x in mapped if x >= 0)
                            padded = sum(
                                ((n + block - 1) // block) * block
                                for n in counts.values()
                            )
                            # Verify materialized tables at boundaries; large prefill uses counts only.
                            if tokens <= 257 and seed == 0:
                                packed, experts = pack_reference(mapped, local, block)
                                verify_pack(mapped, packed, experts, block)
                                verified += 1
                            records.append(
                                dict(
                                    ep=ep,
                                    tokens=tokens,
                                    pattern=pattern,
                                    seed=seed,
                                    rank=rank,
                                    block=block,
                                    active_routes=sum(counts.values()),
                                    active_experts=len(counts),
                                    active_padded_rows=padded,
                                    active_blocks=padded // block,
                                    inactive_routes=sum(x < 0 for x in mapped),
                                    baseline_block=choose_block(
                                        tokens, topk, local, global_experts, "baseline"
                                    ),
                                    ep_density_block=choose_block(
                                        tokens,
                                        topk,
                                        local,
                                        global_experts,
                                        "ep_density",
                                    ),
                                )
                            )
    with (out / "routing_geometry.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = dict(
        status="passed",
        scope="synthetic CPU routing geometry; no GEMM or timing evidence",
        rows=len(records),
        materialized_tables_verified=verified,
        seeds=seeds,
        active_candidate_modified=False,
        performance_winner=None,
        ep8_example={
            str(m): {
                p: choose_block(m, 8, 32, 256, p) for p in ("baseline", "ep_density")
            }
            for m in (8, 32, 128, 256)
        },
        caveats=[
            "Counts do not predict GEMM latency or serving speed.",
            "Fewer padded rows can mean more blocks and weight reloads.",
            "Inspect every rank; a balanced estimate misses hot ranks.",
            "all_remote is relative to rank zero; other ranks still own the work.",
        ],
    )
    (out / "structure_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def cuda_route_checks(out):
    """Optional T4-safe graph test of counts. No SGLang kernels or GEMMs."""
    import torch

    report = {"scope": "synthetic PyTorch mapping/count graph; not SGLang alignment"}
    if not torch.cuda.is_available():
        report.update(status="skipped", reason="CUDA unavailable")
    else:
        report.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__)
        checks = 0
        for tokens in (8, 9, 33, 129, 257):
            ids = torch.zeros((tokens, 8), device="cuda", dtype=torch.int32)
            local = 32
            # Fixed-size equality reduction avoids data-dependent bincount allocations.
            experts = torch.arange(local, device="cuda", dtype=torch.int32)

            def count():
                mapped = torch.where((ids >= 0) & (ids < local), ids, -1)
                counts = (mapped.reshape(-1, 1) == experts).sum(0)
                return mapped, counts

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    count()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                mapped, counts = count()
            for seed in range(2):
                for pattern in PATTERNS:
                    rows = make_routes(tokens, 8, 256, 8, pattern, seed)
                    ids.copy_(torch.tensor(rows, device="cuda", dtype=torch.int32))
                    graph.replay()
                    expected = map_routes(rows, local, 0)
                    assert mapped.reshape(-1).cpu().tolist() == expected
                    expected_counts = Counter(x for x in expected if x >= 0)
                    assert counts.cpu().tolist() == [
                        expected_counts[e] for e in range(local)
                    ]
                    for block in BLOCKS:
                        padded = ((counts + block - 1) // block * block).sum().item()
                        assert padded == sum(
                            ((c + block - 1) // block * block)
                            for c in expected_counts.values()
                        )
                        checks += 1
            del graph
        report.update(status="passed", checks=checks)
    Path(out, "cuda_route_checks.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def manifest(out):
    out = Path(out)
    values = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.iterdir()
        if p.is_file() and p.name != "checksums.json"
    }
    (out / "checksums.json").write_text(json.dumps(values, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("tile_structure_results"))
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--cuda-route-checks", action="store_true")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    print(json.dumps(structure_study(args.out, args.seeds), indent=2))
    if args.cuda_route_checks:
        print(json.dumps(cuda_route_checks(args.out), indent=2))
    manifest(args.out)


if __name__ == "__main__":
    main()
