"""Paired full-K BF16 projection timings. Invoke --cpu for the dispatch fixture."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch
import torch.nn.functional as F

from compact_fc import compact_fc

HERE = Path(__file__).resolve().parent
SHA = "8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
PARENT = HERE.parent / "release_2026-11-18/candidate.diff"
DIM_IN, DIM_OUT, WIDTH = 40960, 8192, 8
STEPS = 12


def make_counts(batch, trial, mode, gen):
    if mode == "full":
        return torch.full((batch,), WIDTH, dtype=torch.int32)
    if trial % 5 == 0:
        return torch.randint(1, WIDTH + 1, (batch,), generator=gen, dtype=torch.int32)
    # Independent near-four rows and dispersed partials include edge cases.
    if trial % 5 == 1:
        return torch.randint(3, 6, (batch,), generator=gen, dtype=torch.int32)
    return torch.randint(1, 8, (batch,), generator=gen, dtype=torch.int32)


def parent_fc(h, w, l):
    return F.linear(h, w)


def control_fc(h, w, l):
    return F.linear(h, w)


def tested_fc(h, w, l):
    return compact_fc(h, w, l, WIDTH)


def assert_cpu():
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == SHA
    assert (
        b"diff --git a/python/sglang/srt/models/dflash.py " not in PARENT.read_bytes()
    )
    assert (
        b"diff --git a/python/sglang/srt/speculative/dflash_worker_v2.py "
        not in PARENT.read_bytes()
    )
    for b in [1, 4, 16, 32]:
        g = torch.Generator().manual_seed(71)
        h = torch.randint(-3, 4, (b * 8, 256), generator=g).bfloat16()
        w = torch.randint(-3, 4, (96, 256), generator=g).bfloat16()
        for t in range(6):
            lengths = make_counts(b, t, "full" if t == 5 else "mixed", g)
            base = parent_fc(h, w, lengths)
            compact = tested_fc(h, w, lengths)
            live = (torch.arange(8) < lengths[:, None]).flatten()
            assert torch.equal(base[live], compact[live])
            if not live.all():
                assert compact[~live].count_nonzero() == 0
    print("CPU source, full-acceptance and variable-prefix fixtures passed.")


def sample(name, weights, batch, seed, mode):
    shape_seed = seed + batch * 13
    rng = random.Random(shape_seed)
    cpu = torch.Generator().manual_seed(shape_seed)
    cuda = torch.Generator(device="cuda").manual_seed(shape_seed)
    times = {k: [] for k in ("baseline", "candidate", "control")}
    gpu_times = {k: [] for k in times}
    states = []
    for t in range(STEPS + 2):
        lengths = make_counts(batch, t, mode, cpu).to("cuda")
        h = torch.randn(
            (batch * 8, DIM_IN), device="cuda", dtype=torch.bfloat16, generator=cuda
        )
        states.append((h, lengths))
    funcs = {"baseline": parent_fc, "control": control_fc, "candidate": tested_fc}
    wrong = total = 0
    worst = 0.0
    differing_trials = 0
    comparisons = []
    order_log = []
    for t, (h, lengths) in enumerate(states):
        if t % 2:
            order = ["candidate", "baseline"]
        else:
            order = ["baseline", "candidate"]
        order.insert(rng.randrange(3), "control")
        order_log.append(order)
        out = {}
        for label in order:
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            torch.cuda.synchronize()
            begin = time.perf_counter_ns()
            start.record()
            out[label] = funcs[label](h, weights, lengths)
            end.record()
            end.synchronize()
            wall = (time.perf_counter_ns() - begin) / 1000
            if t >= 2:
                times[label].append(wall)
                gpu_times[label].append(start.elapsed_time(end) * 1000)
        mask = (torch.arange(WIDTH, device="cuda") < lengths[:, None]).reshape(-1)
        actual = out["candidate"][mask]
        correct = out["baseline"][mask]
        assert torch.equal(out["control"], out["baseline"])
        # Native behavior is the gate for every retained BF16 output bit.
        mismatch = (actual.view(torch.int16) != correct.view(torch.int16)).sum().item()
        wrong += mismatch
        total += correct.numel()
        differing_trials += bool(mismatch)
        worst = max(worst, (actual.float() - correct.float()).abs().max().item())
        comparisons.append(dict(live=int(mask.sum().item()), mismatch=int(mismatch)))
    return (
        {"weight": 1, "times": times},
        {
            "gpu_event_us": gpu_times,
            "paired_order": order_log[2:],
            "mismatching_bf16": wrong,
            "retained_bf16": total,
            "differing_trials": differing_trials,
            "max_abs": worst,
            "per_input": comparisons,
        },
    )


def run(stage):
    seed = 127109 if stage == "pilot" else 873821
    rng = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randn(
        (DIM_OUT, DIM_IN), device="cuda", dtype=torch.bfloat16, generator=rng
    )
    weight.mul_(DIM_IN**-0.5)
    bench = {"instances": 1, "shapes": {}}
    detail = {}
    for batch in (1, 4, 16, 32):
        for mode in ("mixed", "full"):
            name = f"b{batch}_{mode}"
            bench["shapes"][name], detail[name] = sample(
                name, weight, batch, seed, mode
            )
            row = bench["shapes"][name]["times"]
            print(
                stage,
                name,
                round(
                    statistics.median(row["baseline"])
                    / statistics.median(row["candidate"]),
                    3,
                ),
                detail[name]["mismatching_bf16"],
                detail[name]["retained_bf16"],
                flush=True,
            )
    # Captured path must fall back without any dynamic nonzero / shape readbacks.
    x = torch.randn((32, DIM_IN), device="cuda", dtype=torch.bfloat16, generator=rng)
    counts = torch.tensor([1, 2, 7, 5], device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        sampled = tested_fc(x, weight, counts)
    for _ in range(5):
        x.normal_(generator=rng)
        g.replay()
        assert torch.equal(sampled, F.linear(x, weight))
    correct = all(v["mismatching_bf16"] == 0 for v in detail.values())
    result = dict(
        agent="speculation",
        candidate="dflash-prefix-valid-native-bf16",
        scope="single SM86 per-rank full-K native DFlash2 fc, includes eager index readback, packing and scatter; omits downstream RMS, KV, target/draft model",
        baseline_sha256=SHA,
        correct=correct,
        active=True,
        bench=bench,
        stage=stage,
        seed=seed,
        details=detail,
        sources={
            "base": "297e54fd105e974cdf401e5b4d20253a22c16f90",
            "weight_shape": [DIM_OUT, DIM_IN],
        },
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        timing_unit="us wall, synchronized; per-arm CUDA events also recorded",
        cuda_graph_fallback_parity_checks=5,
        allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    )
    (HERE / f"{stage}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("cpu", "pilot", "heldout"))
    stage = p.parse_args().stage
    if stage == "cpu":
        assert_cpu()
    else:
        with torch.inference_mode():
            run(stage)
