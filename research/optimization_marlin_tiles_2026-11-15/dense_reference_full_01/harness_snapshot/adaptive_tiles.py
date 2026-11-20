"""Experimental graph-safe choice of tile 16 only without extra active blocks.

Both alignment tables are built. A device histogram disables the unused GEMM by
publishing zero padded tokens. The two GEMMs share the original output buffer;
exactly one performs active work. There is no host readback or precision change.
"""

import triton
import triton.language as tl


@triton.jit
def _select_tile(
    IDS,
    BIG_COUNT,
    SMALL_COUNT,
    N: tl.constexpr,
    E: tl.constexpr,
    BIG: tl.constexpr,
    NP: tl.constexpr,
    NB: tl.constexpr,
):
    offsets = tl.arange(0, NP)
    ids = tl.load(IDS + offsets, offsets < N, other=-1)
    counts = tl.histogram(ids + 1, NB)
    buckets = tl.arange(0, NB)
    active = (buckets > 0) & (buckets <= E)
    small_blocks = tl.sum(tl.where(active, (counts + 15) // 16, 0), 0)
    big_blocks = tl.sum(tl.where(active, (counts + BIG - 1) // BIG, 0), 0)
    use_small = small_blocks <= big_blocks
    big_total = tl.load(BIG_COUNT)
    small_total = tl.load(SMALL_COUNT)
    tl.store(BIG_COUNT, tl.where(use_small, 0, big_total))
    tl.store(SMALL_COUNT, tl.where(use_small, small_total, 0))


def prepare_adaptive_alignment(local_ids, big_block, local_experts, big_count, is_ep):
    if not is_ep or big_block <= 16 or local_ids.numel() > 8192 or local_experts > 1024:
        return None
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

    small = moe_align_block_size(local_ids, 16, local_experts)
    _select_tile[(1,)](
        local_ids,
        big_count,
        small[2],
        local_ids.numel(),
        local_experts,
        big_block,
        triton.next_power_of_2(local_ids.numel()),
        triton.next_power_of_2(local_experts + 1),
    )
    return small


def adaptive_gemm(*args, adaptive_tables=None, **kwargs):
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm

    output = moe_wna16_marlin_gemm(*args, **kwargs)
    if adaptive_tables is None:
        return output
    alternate = list(args)
    # The public positional API: c at 1; sorted IDs, block IDs, padded count at 10:13.
    assert alternate[1] is not None
    alternate[10:13] = adaptive_tables
    kwargs = dict(kwargs, moe_block_size=16)
    return moe_wna16_marlin_gemm(*alternate, **kwargs)
