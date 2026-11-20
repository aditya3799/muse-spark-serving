"""Paired GPU replay of the pinned draft-KV path with random sparse cache slots."""

import argparse
import json
from pathlib import Path
import random
import statistics
import time

import torch

from benchmark import SHA
from kv_integrated_kernel import _fused_norm_rope_kernel_stacked as fused_direct
from kv_kernels import (
    _fused_norm_rope_kernel_stacked,
    _direct_norm_rope,
    set_kv_buffer_prefix_valid_tiled,
)

HERE = Path(__file__).resolve().parent
LAYERS, DIM, WIDTH, HIDDEN = 5, 128, 8, 8192
CAPACITY = 32768


def pointers(tensors):
    return torch.tensor(
        [t.data_ptr() for t in tensors], dtype=torch.uint64, device="cuda"
    )


def capture(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def run_case(b, mode, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed + b)
    rng = random.Random(seed + b + 17)

    def bf16(*shape):
        return torch.randn(shape, generator=gen, device="cuda", dtype=torch.bfloat16)

    m = b * WIDTH
    h = bf16(m, HIDDEN)
    weight = bf16(HIDDEN, LAYERS * 2 * DIM)
    weight.mul_(HIDDEN**-0.5)
    proj = bf16(m, LAYERS, 2 * DIM)
    norm = bf16(LAYERS, DIM)
    eps = torch.full((LAYERS,), 1.0e-5, device="cuda")
    rope = bf16(131072, DIM)
    pos = torch.empty(m, device="cuda", dtype=torch.int64)
    loc = torch.empty(m, device="cuda", dtype=torch.int64)
    lens = torch.empty(b, device="cuda", dtype=torch.int32)
    temp_k, temp_v = [bf16(LAYERS, m, 1, DIM) for _ in range(2)]

    # Separate layer allocations mirror the actual BF16 pool pointer array.
    def pools():
        k = [
            torch.zeros((CAPACITY + 1, 1, DIM), dtype=h.dtype, device="cuda")
            for _ in range(LAYERS)
        ]
        v = [torch.zeros_like(t) for t in k]
        return k, v, pointers(k), pointers(v)

    base_pool = pools()
    candidate_pool = pools()
    control_pool = pools()

    def call(direct, pool):
        keys, values, k_ptrs, v_ptrs = pool
        # Same actual native BF16 matrix multiply in both arms. The direct
        # change only replaces the materialization/scatter after this point.
        torch.mm(h, weight, out=proj.view(m, -1))
        storek, storev = (k_ptrs, v_ptrs) if direct else (temp_k, temp_v)
        grid = (m, 1, LAYERS)
        args = (
            proj,
            norm,
            eps,
            rope,
            pos,
            storek,
            storev,
            LAYERS * 2 * DIM,
            2 * DIM,
            DIM,
            DIM,
            m * DIM,
            DIM,
            DIM,
            m * DIM,
            DIM,
            DIM,
            m,
            LAYERS,
            1,
            DIM,
            DIM,
            DIM,
            DIM // 2,
            DIM,
        )
        if direct:
            fused_direct[grid](*args, loc, lens, WIDTH)
        else:
            _fused_norm_rope_kernel_stacked[grid](*args)
            for layer in range(LAYERS):
                set_kv_buffer_prefix_valid_tiled[(b, WIDTH, 2)](
                    temp_k[layer],
                    temp_v[layer],
                    keys[layer],
                    values[layer],
                    loc,
                    lens,
                    DIM * 2,
                    DIM * 2,
                    DIM * 2,
                    DIM * 2,
                    WIDTH,
                    ROW_BYTES=DIM * 2,
                    BYTES_PER_TILE=128,
                    num_warps=4,
                    num_stages=2,
                )

    loc.copy_(torch.arange(m, device="cuda") + 1)
    pos.zero_()
    lens.fill_(4)
    arms = {
        "baseline": lambda: call(False, base_pool),
        "candidate": lambda: call(True, candidate_pool),
        "control": lambda: call(False, control_pool),
    }
    graphs = {name: capture(fn) for name, fn in arms.items()}
    times = {k: [] for k in arms}
    walls = {k: [] for k in arms}
    checks = 0
    order_log = []
    history = []
    mismatch = 0
    for step in range(26):
        h.normal_(generator=gen)
        loc.copy_(torch.randperm(CAPACITY, generator=gen, device="cuda")[:m] + 1)
        pos.copy_(torch.randint(40000, 110000, (m,), generator=gen, device="cuda"))
        if mode == "full":
            lens.fill_(8)
        else:
            lens.random_(1, 9, generator=gen)
        order = (
            ["baseline", "candidate"] if step % 2 == 0 else ["candidate", "baseline"]
        )
        order.insert(rng.randrange(3), "control")
        # Exercise separate graph captures against independent live states and
        # verify their physical buffers alongside the regular eager calls.
        if step < 2:
            for graph in graphs.values():
                graph.replay()
            for layer in range(LAYERS):
                for part in (0, 1):
                    graph_ref = base_pool[part][layer][loc].view(torch.int16)
                    assert torch.equal(
                        candidate_pool[part][layer][loc].view(torch.int16), graph_ref
                    )
                    assert torch.equal(
                        control_pool[part][layer][loc].view(torch.int16), graph_ref
                    )
        for key in order:
            torch.cuda.synchronize()
            a, z = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start = time.perf_counter_ns()
            a.record()
            for _ in range(4):
                arms[key]()
            z.record()
            z.synchronize()
            if step >= 2:
                times[key].append(a.elapsed_time(z) * 250)
                walls[key].append((time.perf_counter_ns() - start) / 4000)
        if step >= 2:
            order_log.append(order)
        history.append(lens.tolist())
        # Include changed and unselected positions, then check the entire cache
        # after the final replay for stray or previously overwritten writes.
        ix = loc if step < 25 else slice(None)
        for layer in range(LAYERS):
            for part in (0, 1):
                expected = base_pool[part][layer][ix].view(torch.int16)
                actual = candidate_pool[part][layer][ix].view(torch.int16)
                ctrl = control_pool[part][layer][ix].view(torch.int16)
                assert torch.equal(ctrl, expected)
                mismatch += (expected != actual).sum().item()
                checks += expected.numel()
    return {"weight": 1, "times": walls}, {
        "compared_bits_as_bf16": checks,
        "mismatching_values": mismatch,
        "paired_order": order_log,
        "cuda_event_us": times,
        "commit_lens": history,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["pilot", "heldout"])
    stage = p.parse_args().stage
    seed = 137561 if stage == "pilot" else 788933
    result = dict(
        agent="speculation",
        candidate="dflash-direct-bf16-draft-kv-write",
        scope="single SM86 native 8192x1280 BF16 projection plus exact five-layer RMSNorm/RoPE and prefix-valid draft cache writes, actual out-of-graph eager step plus separate graph coverage, discontiguous real physical indices, one local KV head; excludes upstream 40960x8192 fc, TP, target and runtime",
        baseline_sha256=SHA,
        correct=True,
        active=True,
        bench={"instances": 1, "shapes": {}},
        stage=stage,
        seed=seed,
        details={},
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
    )
    with torch.inference_mode():
        for b in (1, 4, 16, 32):
            for mode in ("mixed", "full"):
                key = f"b{b}_{mode}"
                row, extra = run_case(b, mode, seed)
                result["correct"] &= not extra["mismatching_values"]
                result["bench"]["shapes"][key] = row
                result["details"][key] = extra
                print(
                    stage,
                    key,
                    round(
                        statistics.median(row["times"]["baseline"])
                        / statistics.median(row["times"]["candidate"]),
                        3,
                    ),
                    extra["mismatching_values"],
                    flush=True,
                )
                (HERE / f"kv_{stage}.json").write_text(json.dumps(result, indent=2))
