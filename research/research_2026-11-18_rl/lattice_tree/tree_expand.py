"""Pack per-node ancestor KV slots directly from the selected tree for FA3's shared-prefix verify."""
import triton
import triton.language as tl

@triton.jit
def _expand(slots, masks, depths, seq_len, page_table, expand_len, swa_len, positions,
            N: tl.constexpr):
    row = tl.program_id(0)
    req = row // N
    node = row % N
    col = tl.arange(0,N)
    bits = tl.load(masks + row).to(tl.uint32)
    depth = tl.load(depths + row)
    seq = tl.load(seq_len + req)
    present = ((bits >> col.to(tl.uint32)) & 1) != 0
    count = depth + 1
    rank = tl.cumsum(present.to(tl.int32), 0)
    # The frozen FA3 expand-table path orders valid entries by source index, then appends all others.
    dest = tl.where(present, rank - 1, count + col - rank)
    slot = tl.load(slots + req*N + col)
    tl.store(page_table + row*N + dest, slot)
    tl.store(expand_len+row, count)
    tl.store(swa_len+row, seq+count)
    tl.store(positions+row, (seq+depth).to(tl.int64))
