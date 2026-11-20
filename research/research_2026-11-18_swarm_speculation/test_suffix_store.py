"""CPU execution tests and optional CUDA replay for the request-owned suffix module."""

import collections
import hashlib
import json
import random
from pathlib import Path
import sys
import torch
from tokenizers import Tokenizer
from suffix_store import SuffixStore

HERE = Path(__file__).resolve().parent
SEP = 2**30


def expected(hist, remaining, gate=8):
    for k in (16, 8):
        key = hist[-k:]
        if k < gate or len(key) < k or SEP in key:
            continue
        for end in range(len(hist) - 1, k - 1, -1):
            if hist[end - k : end] == key:
                tail = hist[end : end + 31]
                if len(tail) == 31 and SEP not in tail and remaining >= 32:
                    return [hist[-1]] + tail
                return None
    return None


def query(s, h, seq, remain=64):
    v, ok = s.propose([h.slot], [h.epoch], [len(seq)], [seq[-1]], [remain])
    return v[0].tolist() if ok[0].item() else None


def check(s, v, h, remain=64):
    actual = query(s, h, v, remain)
    oracle = expected(v, remain)
    assert actual is None or actual == oracle
    return actual, oracle


def contracts(device):
    s = SuffixStore(3, 512, buckets=1024, device=device, stop_ids=(SEP,))
    key = list(range(10, 28))
    base = [3, 4] + key + list(range(100, 170)) + key
    h = s.bind(1, "r1", base)
    h2 = s.bind(0, "r2", [1, 2, 3])
    assert check(s, base, h)[0] is not None
    assert query(s, h, base, 8) is None
    pred, ok = s.propose([1, 0], [h.epoch, h2.epoch], [len(base), 3], [44, 3], [50, 50])
    assert not ok.any()
    # Partial counts update only committed output; the old anchor stays once.
    add = s.append_verified([1], [h.epoch], [len(base)], [[3, 4, 5, 20]], [2])
    assert add.item() and s.frontier[1].item() == len(base) + 2
    assert not s.append_verified([1], [h.epoch], [len(base)], [[7]], [1]).item()
    old = h
    h = s.truncate(h, len(base) + 2, len(base))
    assert h and not s.append_verified([1], [old.epoch], [len(base)], [[7]], [1]).item()
    assert check(s, base, h)[0] is not None
    dup = s.append_verified(
        [1, 1], [h.epoch, h.epoch], [len(base)] * 2, [[23], [24]], [1, 1]
    )
    assert not dup.any() and s.frontier[1].item() == len(base)
    delayed = s.append_verified(
        [1, 1], [h.epoch, h.epoch], [len(base), len(base) + 1], [[25], [26]], [1, 1]
    )
    assert not delayed.any() and s.frontier[1].item() == len(base)
    assert not s.append_verified([1], [h.epoch], [len(base)], [[-1]], [1]).item()
    assert not s.append_verified([1], [h.epoch], [len(base)], [[0] * 513], [513]).item()
    assert s.release(h)
    h3 = s.bind(1, "r3", list(reversed(base)))
    stale = s.append_verified([1], [h.epoch], [len(base)], [[77]], [1])
    assert not stale.item() and not s.release(h)
    both = s.append_verified(
        [1, 1], [h.epoch, h3.epoch], [len(base)] * 2, [[77], [88]], [1, 1]
    )
    assert both.tolist() == [False, True]
    assert s.tokens[1, len(base)].item() == 88
    # Force every key into one bucket and verify every reported candidate by
    # direct history equality. Evictions may lower coverage.
    small = SuffixStore(1, 256, buckets=1, device=device)
    collision = small.bind(0, "collide", base)
    assert check(small, base, collision)[0] is None
    # Cross-message stop tokens block a continuation.
    s.release(h3)
    block = [*key, *range(8), SEP, *range(500, 550), *key]
    h4 = s.bind(1, "stop", block)
    assert query(s, h4, block) is None


def traces(device):
    path = HERE.parent / "research_2026-11-18_frontier/muse_tokenizer.json"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "ac2f41ef2444dec82fdc90e334f80543f298535005fe78aec7e8311ef86f84d0"
    )
    tok = Tokenizer.from_file(str(path))
    counts = collections.Counter()
    rng = random.Random(3281)
    for off in (1000, 5000, 12000, 20000):
        row = json.loads((HERE / f"traces_{off}.json").read_text(encoding="utf8"))[
            "rows"
        ][0]["row"]
        msgs = json.loads(row["messages"])
        history = []
        samples = []
        last = {8: {}, 16: {}}
        for m in msgs:
            history.append(SEP)
            tokens = tok.encode(m["content"], add_special_tokens=False).ids
            for p, token in enumerate(tokens):
                n = len(history)
                if m["role"] == "assistant" and len(tokens) - p >= 32 and p >= 8:
                    source = None
                    for k in (16, 8):
                        if p >= k:
                            source = last[k].get(tuple(history[-k:]))
                            if source is not None:
                                break
                    if source is not None or rng.random() < 0.04:
                        if rng.random() < 0.13:
                            samples.append((n, len(tokens) - p))
                for k in (16, 8):
                    if p >= k:
                        last[k][tuple(history[-k:])] = n
                history.append(token)
        if len(samples) > 70:
            samples = sorted(rng.sample(samples, 70))
        s = SuffixStore(
            1, len(history) + 2, buckets=131072, device=device, stop_ids=(SEP,)
        )
        h = s.bind(0, row["instance_id"], [])
        prev = 0
        for n, remaining in samples:
            for start in range(prev, n, 500):
                part = history[start : min(n, start + 500)]
                assert s.append_verified(
                    [0], [h.epoch], [start], [part], [len(part)]
                ).item()
            actual, oracle = check(s, history[:n], h, remaining)
            counts["probes"] += 1
            counts["oracle_hits"] += oracle is not None
            counts["verified_hits"] += actual is not None
            prev = n
    assert counts["probes"] >= 100 and counts["verified_hits"] >= 20
    return dict(counts)


if __name__ == "__main__":
    torch.set_num_threads(1)
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    contracts(device)
    results = traces(device)
    results.update(device=device, failure_modes_passed=True)
    (HERE / f"suffix_store_{device}.json").write_text(json.dumps(results, indent=2))
    print(results)
