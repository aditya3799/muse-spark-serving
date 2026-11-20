"""Select sparse EP tiles on-device while preserving the dense Marlin path.

The caller supplies sorted, already-local routing tables with inactive
blocks first, caller-owned output buffers, and masked inactive output
consumers. The selector preserves every active route and disables the
unused projection with a zero padded count; it never reads back to CPU.
This experimental helper is used only by the Muse Spark Marlin path.
"""

import triton


import triton.language as tl


@triton.jit
def _select_and_compact(
    IDS,
    BIG_TOKENS,
    BIG_COUNT,
    SMALL_TOKENS,
    SMALL_EXPERTS,
    SMALL_COUNT,
    N: tl.constexpr,
    E: tl.constexpr,
    BIG: tl.constexpr,
    SMALL: tl.constexpr,
    NP: tl.constexpr,
    NB: tl.constexpr,
):
    offsets = tl.arange(0, NP)
    ids = tl.load(IDS + offsets, offsets < N, other=-1)
    # Mask padding out of the histogram: it is not a real inactive route.
    ids = tl.where(offsets < N, ids + 1, NB)
    counts = tl.histogram(ids, NB)
    buckets = tl.arange(0, NB)
    active = (buckets > 0) & (buckets <= E) & (counts > 0)
    use_small = tl.max(tl.where(active, counts, 0), 0) <= SMALL
    big_blocks = (counts + BIG - 1) // BIG
    big_starts = tl.cumsum(big_blocks, 0) - big_blocks
    small_starts = tl.cumsum(active.to(tl.int32), 0) - active.to(tl.int32)
    rows = tl.arange(0, SMALL)
    tokens = tl.load(
        BIG_TOKENS + big_starts[:, None] * BIG + rows[None, :],
        use_small & active[:, None],
        other=N,
    )
    tl.store(
        SMALL_TOKENS + small_starts[:, None] * SMALL + rows[None, :],
        tokens,
        use_small & active[:, None],
    )
    tl.store(SMALL_EXPERTS + small_starts, buckets - 1, use_small & active)
    total = tl.load(BIG_COUNT)
    tl.store(BIG_COUNT, tl.where(use_small, 0, total))
    tl.store(
        SMALL_COUNT, tl.where(use_small, tl.sum(active.to(tl.int32), 0) * SMALL, 0)
    )


def prepare_adaptive_alignment(
    local_ids, big_block, local_experts, big_count, is_ep, big_sorted, small_block=16
):
    if (
        not is_ep
        or big_block <= small_block
        or local_ids.numel() > 8192
        or local_experts > 128
    ):
        return None
    import torch

    small = (
        torch.empty(
            local_experts * small_block, dtype=torch.int32, device=local_ids.device
        ),
        torch.empty(local_experts, dtype=torch.int32, device=local_ids.device),
        torch.empty(1, dtype=torch.int32, device=local_ids.device),
    )
    _select_and_compact[(1,)](
        local_ids,
        big_sorted,
        big_count,
        *small,
        local_ids.numel(),
        local_experts,
        big_block,
        small_block,
        triton.next_power_of_2(local_ids.numel()),
        triton.next_power_of_2(local_experts + 1),
    )
    return small


def adaptive_gemm(*args, adaptive_tables=None, adaptive_block=16, **kwargs):
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm

    output = moe_wna16_marlin_gemm(*args, **kwargs)
    if adaptive_tables is None:
        return output
    alternate = list(args)
    # The public positional API: c at 1; sorted IDs, block IDs, padded count at 10:13.
    assert alternate[1] is not None
    alternate[10:13] = adaptive_tables
    kwargs = dict(kwargs, moe_block_size=adaptive_block)
    return moe_wna16_marlin_gemm(*alternate, **kwargs)
