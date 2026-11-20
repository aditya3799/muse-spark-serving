"""Fused CUDA side of the causal suffix store. CPU behavior is in suffix_store."""

import torch
import triton
import triton.language as tl

from suffix_store import SuffixStore


@triton.jit
def _state(
    slots,
    epochs,
    required,
    live,
    frontier,
    row,
    B: tl.constexpr,
    ROWS: tl.constexpr,
    SLOTS: tl.constexpr,
):
    slot = tl.load(slots + row)
    epoch = tl.load(epochs + row)
    n = tl.load(required + row)
    safe = tl.minimum(tl.maximum(slot, 0), SLOTS - 1)
    current = tl.load(live + safe)
    actual = tl.load(frontier + safe)
    rows = tl.arange(0, ROWS)
    other = tl.load(slots + rows, rows < B, -1)
    other_epoch = tl.load(epochs + rows, rows < B, 0)
    # Reject equal-generation duplicate slots even when their frontiers differ.
    # Same slot, epoch and frontier means another writer owns the same row.
    copies = tl.sum(((other == safe) & (other_epoch == current)).to(tl.int32), 0)
    valid = (
        (slot == safe)
        & (epoch > 0)
        & (epoch == current)
        & (n == actual)
        & (copies == 1)
    )
    return safe, n, valid


@triton.jit
def _key_end(
    tokens, ends, stops, n, K: tl.constexpr, NBUCKETS: tl.constexpr, NSTOP: tl.constexpr
):
    h = tl.full((), 0, tl.int64)
    clean = tl.full((), True, tl.int1)
    for u in tl.static_range(K):
        cur = tl.load(tokens + n - K + u, mask=n >= K, other=0)
        h = h * 1000003 + cur.to(tl.int64) + 1
        for z in tl.static_range(NSTOP):
            clean = clean & (cur != tl.load(stops + z))
    bucket = (h ^ (h >> 27)) & (NBUCKETS - 1)
    end = tl.load(ends + bucket).to(tl.int64)
    known = (end >= K) & (end < n) & (n >= K)
    for u in tl.static_range(K):
        cur = tl.load(tokens + n - K + u, mask=n >= K, other=0)
        prev = tl.load(tokens + end - K + u, mask=known, other=0)
        clean = clean & (cur == prev)
    return tl.where(known & clean, end, -1)


@triton.jit
def _lookup(
    tokens,
    ends,
    live,
    frontier,
    slots,
    epochs,
    required,
    anchors,
    remaining,
    stops,
    proposed,
    hit_flag,
    B: tl.constexpr,
    ROWS: tl.constexpr,
    SLOTS: tl.constexpr,
    CAP: tl.constexpr,
    NBUCKETS: tl.constexpr,
    NSTOP: tl.constexpr,
    WIDTH: tl.constexpr,
    PAD: tl.constexpr,
    MIN_KEY: tl.constexpr,
):
    row = tl.program_id(0)
    slot, expected, valid = _state(
        slots, epochs, required, live, frontier, row, B, ROWS, SLOTS
    )
    n = tl.maximum(0, tl.minimum(expected, CAP))
    history = tokens + slot * (CAP + 1)
    tables = ends + slot * 2 * (NBUCKETS + 1)
    anchor = tl.load(anchors + row)
    left = tl.load(remaining + row)
    current = tl.load(history + n - 1, mask=n > 0, other=-1)
    valid = valid & (current == anchor) & (left >= WIDTH)
    if MIN_KEY <= 8:
        short_end = _key_end(history, tables, stops, n, 8, NBUCKETS, NSTOP)
    else:
        short_end = tl.full((), -1, tl.int64)
    long_end = _key_end(history, tables + NBUCKETS + 1, stops, n, 16, NBUCKETS, NSTOP)
    end = tl.where(long_end >= 0, long_end, short_end)
    valid = valid & (end >= 0) & (end + WIDTH - 1 <= n)
    col = tl.arange(0, PAD)
    part = tl.load(
        history + end + col - 1, mask=valid & (col > 0) & (col < WIDTH), other=0
    )
    for z in tl.static_range(NSTOP):
        stop = tl.load(stops + z)
        valid = valid & (
            tl.sum(((part == stop) & (col > 0) & (col < WIDTH)).to(tl.int32), 0) == 0
        )
    result = tl.where(col == 0, anchor, part)
    tl.store(proposed + row * WIDTH + col, tl.where(valid, result, 0), mask=col < WIDTH)
    tl.store(hit_flag + row, valid)


@triton.jit
def _add_keys(
    tokens,
    ends,
    outputs,
    n,
    positions,
    keep,
    K: tl.constexpr,
    NBUCKETS: tl.constexpr,
    HAS_OUTPUT: tl.constexpr,
):
    indexable = keep & (positions >= K)
    h = tl.full(positions.shape, 0, tl.int64)
    for u in tl.static_range(K):
        source = positions - K + u
        if HAS_OUTPUT:
            old = tl.load(tokens + source, indexable & (source < n), other=0)
            new = tl.load(outputs + source - n, indexable & (source >= n), other=0)
            part = tl.where(source < n, old, new)
        else:
            part = tl.load(tokens + source, indexable, other=0)
        h = h * 1000003 + part.to(tl.int64) + 1
    bucket = (h ^ (h >> 27)) & (NBUCKETS - 1)
    tl.atomic_max(ends + bucket, positions.to(tl.int32), mask=indexable, sem="relaxed")


@triton.jit
def _build_index(
    tokens, ends, n, slot, CAP: tl.constexpr, NBUCKETS: tl.constexpr, TILE: tl.constexpr
):
    positions = tl.program_id(0) * TILE + tl.arange(0, TILE)
    history = tokens + slot * (CAP + 1)
    tables = ends + slot * (2 * (NBUCKETS + 1))
    _add_keys(history, tables, tokens, 0, positions, positions < n, 8, NBUCKETS, False)
    _add_keys(
        history,
        tables + NBUCKETS + 1,
        tokens,
        0,
        positions,
        positions < n,
        16,
        NBUCKETS,
        False,
    )


@triton.jit
def _append(
    tokens,
    ends,
    live,
    frontier,
    slots,
    epochs,
    required,
    outputs,
    counts,
    result,
    B: tl.constexpr,
    ROWS: tl.constexpr,
    SLOTS: tl.constexpr,
    CAP: tl.constexpr,
    NBUCKETS: tl.constexpr,
    WIDTH: tl.constexpr,
    PAD: tl.constexpr,
):
    row = tl.program_id(0)
    slot, n, valid = _state(
        slots, epochs, required, live, frontier, row, B, ROWS, SLOTS
    )
    count = tl.load(counts + row)
    col = tl.arange(0, PAD)
    part = tl.load(outputs + row * WIDTH + col, col < WIDTH, other=0)
    used = col < count
    invalid_id = used & ((part < 0) | (part >= 2147483648))
    valid = (
        valid
        & (count >= 0)
        & (count <= WIDTH)
        & (n + count <= CAP)
        & (tl.sum(invalid_id.to(tl.int32), 0) == 0)
    )
    history = tokens + slot * (CAP + 1)
    tables = ends + slot * (2 * (NBUCKETS + 1))
    _add_keys(
        history,
        tables,
        outputs + row * WIDTH,
        n,
        n + col,
        valid & used,
        8,
        NBUCKETS,
        True,
    )
    _add_keys(
        history,
        tables + NBUCKETS + 1,
        outputs + row * WIDTH,
        n,
        n + col,
        valid & used,
        16,
        NBUCKETS,
        True,
    )
    tl.store(history + n + col, part.to(tl.int32), valid & used)
    tl.store(frontier + slot, n + count, valid)
    tl.store(result + row, valid)


class CudaSuffixStore(SuffixStore):
    def __init__(self, slots, capacity, *, buckets=65536, device="cuda", stop_ids=()):
        super().__init__(
            slots, capacity, buckets=buckets, device=device, stop_ids=stop_ids
        )
        self.stop_buffer = self._ids(stop_ids)

    def _reindex(self, slot, n):
        if n:
            _build_index[(triton.cdiv(n, 128),)](
                self.tokens, self.ends, n, slot, self.capacity, self.buckets, 128
            )

    def _vectors(self, *args):
        vectors = [self._ids(v).contiguous() for v in args]
        if any(v.ndim != 1 or v.shape != vectors[0].shape for v in vectors):
            raise ValueError("need aligned input vectors")
        return vectors

    def propose(
        self, slots, epochs, expected, anchors, remaining, *, width=32, min_suffix=8
    ):
        if width < 2 or min_suffix not in self.suffixes:
            raise ValueError("unsupported width or suffix length")
        s, e, n, a, r = self._vectors(slots, epochs, expected, anchors, remaining)
        b = s.numel()
        outputs = torch.empty((b, width), device=self.device, dtype=torch.int64)
        hits = torch.empty(b, device=self.device, dtype=torch.bool)
        if b:
            _lookup[(b,)](
                self.tokens,
                self.ends,
                self.live,
                self.frontier,
                s,
                e,
                n,
                a,
                r,
                self.stop_buffer,
                outputs,
                hits,
                b,
                triton.next_power_of_2(b),
                self.slots,
                self.capacity,
                self.buckets,
                len(self.stop_ids),
                width,
                triton.next_power_of_2(width),
                min_suffix,
            )
        return outputs, hits

    def append_verified(self, slots, epochs, expected, output, counts):
        s, e, n, c = self._vectors(slots, epochs, expected, counts)
        b = s.numel()
        out = self._ids(output).contiguous()
        if out.ndim != 2 or out.shape[0] != b:
            raise ValueError("need a rectangular output")
        if out.shape[1] < 1 or out.shape[1] > 64:
            return super().append_verified(s, e, n, out, c)
        valid = torch.empty(b, dtype=torch.bool, device=self.device)
        if b:
            width = out.shape[1]
            _append[(b,)](
                self.tokens,
                self.ends,
                self.live,
                self.frontier,
                s,
                e,
                n,
                out,
                c,
                valid,
                b,
                triton.next_power_of_2(b),
                self.slots,
                self.capacity,
                self.buckets,
                width,
                triton.next_power_of_2(width),
            )
        return valid
