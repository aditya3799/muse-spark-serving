"""Experimental shared routing preparation; no weight or arithmetic changes."""
import torch
import triton
import triton.language as tl


@triton.jit
def _counts_and_offsets(ids_ptr, counts_ptr, offsets_ptr,
                        ROUTES: tl.constexpr, EXPERTS: tl.constexpr,
                        CUTOFF: tl.constexpr, BLOCK: tl.constexpr,
                        BINS: tl.constexpr, BLOCK_SIZES: tl.constexpr,
                        IS_LIGHT: tl.constexpr):
    r = tl.arange(0, BLOCK)
    ids = tl.load(ids_ptr + r, mask=r < ROUTES, other=-1)
    valid = (r < ROUTES) & (ids >= 0) & (ids < EXPERTS)
    counts = tl.histogram(tl.where(valid, ids, BINS - 1).to(tl.int32), BINS)
    e = tl.arange(0, BINS)
    counts = tl.where(e < EXPERTS, counts, 0)
    tl.store(counts_ptr + e, counts, mask=e < EXPERTS)
    for plan in tl.static_range(len(BLOCK_SIZES)):
        if IS_LIGHT[plan]:
            selected = counts <= CUTOFF
        else:
            selected = counts > CUTOFF
        bm: tl.constexpr = BLOCK_SIZES[plan]
        padded = tl.where(selected, tl.cdiv(counts, bm) * bm, 0)
        offsets = tl.cumsum(padded) - padded
        tl.store(offsets_ptr + plan * (EXPERTS + 1) + e, offsets, mask=e < EXPERTS)
        tl.store(offsets_ptr + plan * (EXPERTS + 1) + EXPERTS, tl.sum(padded, 0))


@triton.jit
def _scatter_plans(ids_ptr, counts_ptr, offsets_ptr, sorted_ptrs, expert_ptrs,
                   ROUTES: tl.constexpr, EXPERTS: tl.constexpr,
                   CUTOFF: tl.constexpr, BLOCK: tl.constexpr,
                   BLOCK_SIZES: tl.constexpr, IS_LIGHT: tl.constexpr):
    expert = tl.program_id(0)
    plan_id = tl.program_id(1)
    count = tl.load(counts_ptr + expert)
    for plan in tl.static_range(len(BLOCK_SIZES)):
        if plan_id == plan:
            if IS_LIGHT[plan]:
                selected = (count > 0) & (count <= CUTOFF)
            else:
                selected = count > CUTOFF
            if selected:
                bm: tl.constexpr = BLOCK_SIZES[plan]
                offset = tl.load(offsets_ptr + plan * (EXPERTS + 1) + expert)
                r = tl.arange(0, BLOCK)
                ids = tl.load(ids_ptr + r, mask=r < ROUTES, other=-1)
                belongs = (r < ROUTES) & (ids == expert)
                ordinal = tl.cumsum(belongs.to(tl.int32)) - 1
                # Original route slots remain unchanged; each is written once.
                tl.store(sorted_ptrs[plan] + offset + ordinal, r, mask=belongs)
                padded = tl.cdiv(count, bm) * bm
                # Padding belongs to this expert's private interval.
                tl.store(sorted_ptrs[plan] + offset + r, ROUTES,
                         mask=(r >= count) & (r < padded))
                tl.store(expert_ptrs[plan] + offset // bm + r, expert,
                         mask=r < tl.cdiv(count, bm))


def fused_partition_align(local_ids, experts, cutoff, block_sizes):
    """Return {(partition, blockM): (sorted IDs, expert IDs, padded count)}.

    block_sizes lists unique ('light'/'heavy', blockM) pairs. Arrays beyond
    padded count are unspecified; the pinned Humming scheduler never reads them.
    """
    routes = local_ids.numel()
    if (not local_ids.is_cuda or not local_ids.is_contiguous()
            or local_ids.dtype not in (torch.int32, torch.int64)
            or not 0 < routes <= 4096 or not 0 < experts <= 256 or cutoff <= 0):
        raise ValueError('Expected contiguous CUDA IDs, 1..4096 routes, and 1..256 experts')
    keys = tuple(block_sizes)
    if not keys or len(keys) > 4 or len(set(keys)) != len(keys):
        raise ValueError('Expected one to four unique partition/block pairs')
    if any(name not in ('light', 'heavy') or not isinstance(bm, int)
           or isinstance(bm, bool) or not 0 < bm <= 256 or bm % 4 for name, bm in keys):
        raise ValueError('Expected valid partition names and block sizes divisible by four')
    counts = torch.empty((experts,), device=local_ids.device, dtype=torch.int32)
    offsets = torch.empty((len(keys), experts + 1), device=local_ids.device, dtype=torch.int32)
    sorted_ids = []
    expert_ids = []
    for _, bm in keys:
        capacity = triton.cdiv(routes + experts * (bm - 1), bm) * bm
        sorted_ids.append(torch.empty((capacity,), device=local_ids.device, dtype=torch.int32))
        expert_ids.append(torch.empty((capacity // bm,), device=local_ids.device, dtype=torch.int32))
    blocks = tuple(bm for _, bm in keys)
    light = tuple(name == 'light' for name, _ in keys)
    _counts_and_offsets[(1,)](
        local_ids, counts, offsets, ROUTES=routes, EXPERTS=experts,
        CUTOFF=cutoff, BLOCK=triton.next_power_of_2(routes),
        BINS=triton.next_power_of_2(experts + 1), BLOCK_SIZES=blocks,
        IS_LIGHT=light, num_warps=8 if routes > 1024 else 4)
    # A single expert may own every route; padding can exceed ROUTES by bm-1.
    scatter_block = triton.next_power_of_2(routes + max(blocks) - 1)
    _scatter_plans[(experts, len(keys))](
        local_ids, counts, offsets, tuple(sorted_ids), tuple(expert_ids),
        ROUTES=routes, EXPERTS=experts, CUTOFF=cutoff, BLOCK=scatter_block,
        BLOCK_SIZES=blocks, IS_LIGHT=light, num_warps=8 if scatter_block > 1024 else 4)
    return {key: (sorted_ids[i], expert_ids[i], offsets[i, experts:experts + 1])
            for i, key in enumerate(keys)}, counts
