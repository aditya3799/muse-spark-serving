"""Replay packed target verification against the pinned dense CPU reference."""
import ast
import json
from pathlib import Path
from typing import Tuple
import random
import torch
from ragged_accept import GreedyCommit

HERE = Path(__file__).resolve().parent
PIN = HERE.parent / "sglang/python/sglang"


def pinned(path, name):
    tree = ast.parse((PIN / path).read_text())
    fn, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    module = ast.Module(body=[fn], type_ignores=[])
    namespace = dict(torch=torch, Tuple=Tuple)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


accept = pinned("kernels/ops/speculative/dflash.py", "compute_dflash_correct_drafts_and_bonus")
commit = pinned("srt/speculative/dflash_worker_v2.py", "_commit_accept")


def main():
    rng = random.Random(5513)
    torch.manual_seed(5513)
    size, stride = 32, 32
    cap = size * stride
    dev = "cuda"
    ids, pred, slots = [torch.empty(cap, dtype=torch.int64, device=dev) for _ in range(3)]
    starts = torch.empty(size + 1, dtype=torch.int32, device=dev)
    prefix = torch.empty(size, dtype=torch.int64, device=dev)
    out = GreedyCommit.allocate(size, stride, dev)
    starts.copy_(torch.arange(size + 1, dtype=torch.int32))
    ids.zero_()
    pred.zero_()
    slots.zero_()
    prefix.zero_()
    out.verify(ids, pred, starts, prefix, slots)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.verify(ids, pred, starts, prefix, slots)
    checks = 0
    mismatches = set()
    for trial in range(40):
        lens = ([32] * size if trial == 0 else [8] * size if trial == 1 else
                [rng.choice([1, 4, 8, 16, 32]) for _ in range(size)])
        ptr = torch.tensor([0, *list(torch.tensor(lens).cumsum(0).tolist())])
        used = int(ptr[-1])
        x = torch.randint(1, 131072, (cap,))
        y = x.roll(-1)
        # Each row's final prediction is independent of the next row's anchor.
        prev = torch.randint(1, 150000, (size,))
        physical = torch.randperm(cap) + 9
        want = [torch.zeros_like(v, device="cpu") for v in
                (out.lengths, out.bonus, out.new_prefix, out.emitted, out.sources, out.retained_slots)]
        want[4].fill_(-1)
        want[5].fill_(-1)
        for r, width in enumerate(lens):
            a, b = int(ptr[r]), int(ptr[r+1])
            pos = r % width if trial <= 1 else rng.randrange(width)
            mismatches.add((width, pos))
            y[b-1] = rng.randint(0, 131071)
            if pos + 1 < width:
                y[a+pos] = (int(x[a+pos+1]) + 1) % 131072
            draft = x[a:b].reshape(1,-1)
            accepted, bonus = accept(candidates=draft, target_predict=y[a:b].reshape(1,-1))
            output, lengths = commit(draft, accepted, bonus)
            n = int(lengths.item())
            assert n == pos+1
            want[0][r] = n
            want[1][r] = bonus.item()
            want[2][r] = prev[r] + n
            want[3][r, :n] = output[0, :n]
            want[4][r, :n] = torch.arange(a, a+n)
            want[5][r, :n] = physical[a:a+n]
            checks += 1
        ids.copy_(x)
        pred.copy_(y)
        prefix.copy_(prev)
        slots.copy_(physical)
        starts.copy_(ptr)
        graph.replay()
        got = [v.cpu() for v in (out.lengths, out.bonus, out.new_prefix, out.emitted, out.sources, out.retained_slots)]
        assert all(torch.equal(u, v) for u,v in zip(got, want))
        live = got[4] >= 0
        # Only committed input slots may modify the simulated draft KV pool.
        expected = torch.full((cap + 18,), -77)
        actual = expected.clone()
        for r in range(size):
            a, n = int(ptr[r]), int(want[0][r])
            expected[physical[a:a+n]] = torch.arange(a,a+n)
        actual[got[5][live]] = got[4][live].long()
        assert torch.equal(actual, expected)
        assert torch.max(got[4]) < used
        assert not torch.equal(got[4][live] + 1, want[4][live])
    assert all((width,i) in mismatches for width in (1,4,8,16,32) for i in range(width))
    data = dict(requests_checked=checks, cuda_graph_replays=40, device=torch.cuda.get_device_name(),
                widths=[1,4,8,16,32], comparison="pinned dense reference for all live emitted tokens and bonus",
                kv_scope="source and physical-slot ownership with canaries; scheduler allocation, stop truncation, TP and KV writes excluded")
    (HERE/"ragged_accept_qualification.json").write_text(json.dumps(data, indent=2))
    print(json.dumps(data))


if __name__ == "__main__":
    main()

