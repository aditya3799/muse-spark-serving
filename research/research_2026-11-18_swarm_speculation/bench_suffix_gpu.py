"""Measure the prototype on real 74k-token prefixes, changing live graph inputs."""

import collections
import json
import math
from pathlib import Path
import random
import time
import sys

import torch
from tokenizers import Tokenizer

from suffix_routing import DFlashCopyRouter
from suffix_store import SuffixStore

HERE = Path(__file__).resolve().parent
MARKER = 2**30
STEPS = 80


def quantiles(samples):
    ordered = sorted(samples)

    def pct(frac):
        return round(ordered[max(0, math.ceil(len(ordered) * frac) - 1)], 3)

    return {"p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99), "max": ordered[-1]}


def read_trace():
    tokenizer = Tokenizer.from_file(
        str(HERE.parent / "research_2026-11-18_frontier/muse_tokenizer.json")
    )
    rows = json.loads((HERE / "traces_12000.json").read_text(encoding="utf8"))["rows"]
    for entry in rows:
        rows_tokens = []
        for msg in json.loads(entry["row"]["messages"]):
            rows_tokens.append(MARKER)
            rows_tokens.extend(
                tokenizer.encode(msg["content"], add_special_tokens=False).ids
            )
        if len(rows_tokens) >= 79_000:
            return entry["row"]["instance_id"], rows_tokens
    raise ValueError("74k held-out trace missing")


def measure(batch, trace, eager=False):
    n = [74000 + i * 13 for i in range(batch)]
    store = SuffixStore(batch, 131072, buckets=65536, stop_ids=[MARKER], device="cuda")
    router = DFlashCopyRouter(store)
    bind_ms = []
    handles = []
    input_ids = torch.tensor(trace, device="cuda", dtype=torch.int32)
    for i in range(batch):
        torch.cuda.synchronize()
        start = time.perf_counter()
        handles.append(store.bind(i, "trace" + str(i), input_ids[: n[i]]))
        torch.cuda.synchronize()
        bind_ms.append((time.perf_counter() - start) * 1e3)
    slots = torch.arange(batch, device="cuda")
    epochs = torch.tensor([h.epoch for h in handles], device="cuda")
    lens = torch.tensor(n, device="cuda") - 1
    anchors = input_ids[lens].long().clone()
    remain = torch.full_like(lens, 1000)
    eligible = torch.ones_like(lens, dtype=torch.bool)
    counts = torch.zeros_like(lens)
    emitted = torch.zeros((batch, 32), device="cuda", dtype=torch.int64)

    def lookup():
        return router.route(slots, epochs, lens, anchors, remain, eligible)

    def append():
        return router.accept(slots, epochs, lens, emitted, counts)

    for _ in range(2):
        found = lookup()
        valid = append()
    torch.cuda.synchronize()
    if not eager:
        lookup_graph = torch.cuda.CUDAGraph()
        append_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(lookup_graph):
            found = lookup()
        with torch.cuda.graph(append_graph):
            valid = append()

    rng = random.Random(88)
    history_index = {k: collections.defaultdict(list) for k in (8, 16)}
    for k, index in history_index.items():
        for j in range(k, len(trace)):
            index[tuple(trace[j - k : j])].append(j)
    from bisect import bisect_left

    def agrees(history_end, proposal):
        for k, index in history_index.items():
            key = tuple(trace[history_end - k : history_end])
            if MARKER in key:
                continue
            candidates = index.get(key, [])
            pos = bisect_left(candidates, history_end) - 1
            if pos >= 0:
                end = candidates[pos]
                if end + 31 <= history_end and MARKER not in trace[end : end + 31]:
                    if proposal == [trace[history_end - 1], *trace[end : end + 31]]:
                        return True
        return False

    events = []
    hits = 0
    committed = 0
    widths = [1, 2, 3, 3, 4, 4, 4, 5, 5, 8, 16, 32]
    new_lens = []
    next_out = []
    next_ct = []
    for step in range(STEPS):
        count = [rng.choice(widths) for _ in n]
        ids = [trace[p : p + c] + [0] * (32 - c) for p, c in zip(n, count)]
        new_lens.append(n[:])
        next_out.append(ids)
        next_ct.append(count)
        n = [p + c for p, c in zip(n, count)]
    stage_prefix = torch.tensor(new_lens, device="cuda") - 1
    stage_output = torch.tensor(next_out, device="cuda")
    stage_counts = torch.tensor(next_ct, device="cuda")
    stage_anchor = input_ids[stage_prefix].long()
    first = torch.cuda.Event(enable_timing=True)
    first.record()
    first.synchronize()
    for step in range(STEPS):
        lens.copy_(stage_prefix[step])
        anchors.copy_(stage_anchor[step])
        emitted.copy_(stage_output[step])
        counts.copy_(stage_counts[step])
        start, mid, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        start.record()
        if eager:
            found = lookup()
        else:
            lookup_graph.replay()
        mid.record()
        if eager:
            valid = append()
        else:
            append_graph.replay()
        end.record()
        events.append((start, mid, end))
        check = found.copy_rows.tolist()
        outputs = found.proposals.tolist()
        assert valid.all().item()
        assert store.frontier.tolist() == [
            q + c for q, c in zip(new_lens[step], next_ct[step])
        ]
        for row, hit in enumerate(check):
            if hit:
                assert agrees(new_lens[step][row], outputs[row]), (
                    step,
                    row,
                    new_lens[step][row],
                )
                hits += 1
        committed += sum(next_ct[step])
    torch.cuda.synchronize()
    timings = [
        [(x.elapsed_time(y) * 1000) for x, y in ((a, b), (b, c), (a, c))]
        for a, b, c in events
    ]
    return {
        "B": batch,
        "mode": "eager" if eager else "cuda_graph",
        "steps": STEPS,
        "hits_verified_by_independent_history": hits,
        "commits_verified": committed,
        "lookup_us": quantiles([t[0] for t in timings]),
        "append_us": quantiles([t[1] for t in timings]),
        "combined_us": quantiles([t[2] for t in timings]),
        "initial_bind_ms_per_request": quantiles(bind_ms),
        "index_persistent_bytes_per_request": (
            store.tokens.element_size() * store.tokens.numel()
            + store.ends.element_size() * store.ends.numel()
            + 16 * batch
        )
        // batch,
    }


if __name__ == "__main__":
    torch.set_num_threads(1)
    fused = "--fused" in sys.argv
    if fused:
        from suffix_cuda import CudaSuffixStore as SuffixStore
    trace_id, ids = read_trace()
    mixed = "--mixed" in sys.argv
    crop = 8944 if mixed else 0
    ids = ids[crop:]
    summary = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "trace": trace_id,
        "tokens_in_trace": len(ids),
        "real_tokens_cropped_from_start": crop,
        "runs": [],
    }
    for batch in (1, 4, 16, 32):
        result = measure(batch, ids)
        summary["runs"].append(result)
        print(json.dumps(result), flush=True)
        (
            HERE
            / (
                "gpu_suffix_timings_fused_mixed.json"
                if fused and mixed
                else "gpu_suffix_timings_fused.json"
                if fused
                else "gpu_suffix_timings.json"
            )
        ).write_text(json.dumps(summary, indent=2))
