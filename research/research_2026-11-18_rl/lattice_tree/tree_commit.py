"""Commit a page-one greedy verification tree by permuting the request's owned slot IDs."""
import triton
import triton.language as tl


@triton.jit
def _commit(nodes, parents, predictions, slots, req_ids, prefix_lens, remaining,
            request_table, emitted, counts, bonus, src_at_dest, new_lens,
            table_stride, live, WIDTH: tl.constexpr):
    b = tl.program_id(0)
    if b < live:
        offsets = tl.arange(0, WIDTH)
        entries = b * WIDTH + offsets
        token = tl.load(nodes + entries)
        parent = tl.load(parents + entries)
        target = tl.load(predictions + entries)
        old_slot = tl.load(slots + entries)
        req = tl.load(req_ids + b)
        prefix = tl.load(prefix_lens + b)
        budget = tl.minimum(tl.maximum(tl.load(remaining + b), 0), WIDTH)
        current = tl.full((), 0, tl.int32)
        count = tl.full((), 0, tl.int32)
        last = tl.full((), -1, tl.int64)
        keep_rank = tl.full((WIDTH,), -1, tl.int32)
        outputs = tl.full((WIDTH,), 0, tl.int64)
        while (current >= 0) & (count < budget):
            next_token = tl.sum(tl.where(offsets == current, target, 0), 0)
            keep_rank = tl.where(offsets == current, count, keep_rank)
            outputs = tl.where(offsets == count, next_token, outputs)
            match = (offsets != 0) & (parent == current) & (token == next_token)
            current = tl.max(tl.where(match, offsets, -1), 0)
            last = next_token.to(tl.int64)
            count += 1
        # Rejected slots retain their relative order in the reserved tail.
        rejected = (keep_rank < 0).to(tl.int32)
        destination = tl.where(rejected != 0, count + tl.cumsum(rejected, 0) - 1, keep_rank)
        tl.store(request_table + req.to(tl.int64) * table_stride + prefix + destination, old_slot)
        tl.store(src_at_dest + b * WIDTH + destination, offsets)
        tl.store(emitted + entries, outputs)
        tl.store(counts + b, count)
        tl.store(bonus + b, last)
        tl.store(new_lens + b, prefix + count)
