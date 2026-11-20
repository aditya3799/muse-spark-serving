"""Bounded grouped-masked prototype sharing the scored canonical weights.

Capacity is the full route count, even for duplicate expert IDs. This source is
experimental: no production integration or H200 runtime claim.
"""
from copy import deepcopy
import torch
import triton
import triton.language as tl


@triton.jit
def _pack_map(ids, slots, packed_ids, counts, ROUTES: tl.constexpr,
              CAPACITY: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    r = tl.arange(0, BLOCK)
    value = tl.load(ids + r, mask=r < ROUTES, other=-1)
    belongs = (r < ROUTES) & (value == expert)
    count = tl.sum(belongs.to(tl.int32), 0)
    ordinal = tl.cumsum(belongs.to(tl.int32)) - 1
    tl.store(counts + expert, count)
    tl.store(slots + expert * CAPACITY + ordinal, r, mask=belongs)
    tl.store(slots + expert * CAPACITY + r, ROUTES,
             mask=(r >= count) & (r < CAPACITY))
    tl.store(packed_ids + expert * CAPACITY + r,
             tl.where(r < count, expert, -1), mask=r < CAPACITY)


@triton.jit
def _pack_input_and_pad(x, slots, counts, packed_x, activation,
                        CAPACITY: tl.constexpr, TOPK: tl.constexpr,
                        K: tl.constexpr, I: tl.constexpr,
                        UP_BLOCK: tl.constexpr, DOWN_BLOCK: tl.constexpr,
                        BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    expert = row // CAPACITY
    local_row = row % CAPACITY
    count = tl.load(counts + expert)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    if local_row < tl.cdiv(count, UP_BLOCK) * UP_BLOCK:
        slot = tl.load(slots + row)
        values = tl.load(x + (slot // TOPK) * K + columns,
                         mask=(local_row < count) & (columns < K), other=0)
        tl.store(packed_x + row * K + columns, values, mask=columns < K)
    # The unchanged filtered CUDA activation writes real rows only. Initialize
    # just the padding that the down GEMM's TMA tile can read, before activation.
    if (local_row >= count) & (local_row < tl.cdiv(count, DOWN_BLOCK) * DOWN_BLOCK):
        tl.store(activation + row * I + columns, 0, mask=columns < I)


@triton.jit
def _scatter_output(grouped, slots, packed_ids, output,
                    K: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    expert = tl.load(packed_ids + row)
    if expert >= 0:
        slot = tl.load(slots + row)
        columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        value = tl.load(grouped + row * K + columns, mask=columns < K, other=0)
        tl.store(output + slot * K + columns, value, mask=columns < K)


def prepare_grouped_configs(layer):
    from humming.layer import HummingMethod
    owned = {}
    for stage in ('w13', 'w2'):
        default = HummingMethod.get_default_tuning_configs(
            layer, use_f16_accum=False, gemm_type='grouped_masked', sublayer_name=stage)
        snapshot = deepcopy(default)
        copied = deepcopy(default)
        for _, _, config in copied:
            config['use_stream_k'] = False
            config['use_f16_accum'] = False
        assert default == snapshot
        owned[stage] = copied
    return owned


def grouped_masked_outputs(scored_helper, layer, grouped_configs, x, local_ids,
                           torch, global_experts, *, retain=False,
                           defer_downscale=False, workspace_limit=512 * 1024 * 1024):
    from humming.layer import HummingMethod
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.muse_spark_humming_scale import scale_active_rows
    def fallback():
        return scored_helper(layer, x, local_ids, torch, retain=retain,
                             defer_downscale=defer_downscale, global_experts=global_experts)
    if (x.ndim != 2 or local_ids.ndim != 2 or x.dtype != torch.bfloat16
            or not x.is_cuda or local_ids.device != x.device
            or not x.is_contiguous() or not local_ids.is_contiguous()
            or local_ids.dtype not in (torch.int32, torch.int64)
            or local_ids.shape[0] != x.shape[0]):
        return fallback()
    m, k = x.shape
    routes = local_ids.numel()
    experts = layer.w13_post_scale.numel()
    intermediate = layer.humming_metas['w2'].shape_k
    if not 0 < routes <= 4096 or not 0 < experts <= 256:
        return fallback()
    selected = {stage: next(config for lo, hi, config in intervals if lo < routes <= hi)
                for stage, intervals in grouped_configs.items()}
    if any(config.get('use_stream_k') is not False
           or config.get('use_f16_accum') is not False
           or config.get('multi_cast_size_b', 1) != 1
           for config in selected.values()):
        return fallback()
    up_block = selected['w13']['block_shape'][0]
    down_block = selected['w2']['block_shape'][0]
    capacity = max(triton.cdiv(routes, up_block) * up_block,
                   triton.cdiv(routes, down_block) * down_block)
    packed_rows = experts * capacity
    # Count total temporary tensor bytes, not merely the delta versus baseline.
    # Eight MiB reserve covers per-launch TMA descriptors/allocator granularity;
    # measured graph-pool/driver memory remains a separate promotion gate.
    workspace = (2 * packed_rows * (k + max(2 * intermediate, k) + intermediate)
                 + 2 * routes * k + 8 * packed_rows + 4 * experts + 8 * 1024 * 1024)
    if workspace > workspace_limit:
        return fallback()
    device = x.device
    slots = torch.empty((packed_rows,), dtype=torch.int32, device=device)
    packed_ids = torch.empty_like(slots)
    counts = torch.empty((experts,), dtype=torch.int32, device=device)
    _pack_map[(experts,)](local_ids, slots, packed_ids, counts, ROUTES=routes,
                          CAPACITY=capacity, BLOCK=triton.next_power_of_2(capacity),
                          num_warps=8 if capacity > 1024 else 4)
    packed_x = torch.empty((packed_rows, k), dtype=x.dtype, device=device)
    cache13 = torch.empty((packed_rows * max(2 * intermediate, k),), dtype=x.dtype, device=device)
    cache1 = cache13[:packed_rows * 2 * intermediate].view(packed_rows, 2 * intermediate)
    cache2 = torch.empty((packed_rows, intermediate), dtype=x.dtype, device=device)
    _pack_input_and_pad[(packed_rows, triton.cdiv(max(k, intermediate), 512))](
        x, slots, counts, packed_x, cache2, CAPACITY=capacity,
        TOPK=local_ids.shape[1], K=k, I=intermediate, UP_BLOCK=up_block,
        DOWN_BLOCK=down_block, BLOCK_N=512)
    common = dict(expert_layout=counts, top_k=1, valid_shape_m=routes,
                  compute_config={'gemm_type': 'grouped_masked', 'use_f16_accum': False})
    HummingMethod.forward_layer(layer, inputs=packed_x, outputs=cache1,
                                sublayer_name='w13', tuning_config=selected['w13'], **common)
    # Preserve both BF16 rounding boundaries and the original CUDA SiLU.
    scale_active_rows(cache1, packed_ids, layer.w13_post_scale)
    silu_and_mul(cache1, cache2, expert_ids=packed_ids, expert_step=1)
    cache3 = cache13[:packed_rows * k].view(packed_rows, k)
    HummingMethod.forward_layer(layer, inputs=cache2, outputs=cache3,
                                sublayer_name='w2', tuning_config=selected['w2'], **common)
    output = torch.empty((routes, k), dtype=x.dtype, device=device)
    _scatter_output[(packed_rows, triton.cdiv(k, 512))](cache3, slots, packed_ids, output,
                                                       K=k, BLOCK_N=512)
    if not defer_downscale:
        scale_active_rows(output, local_ids, layer.w2_post_scale)
    return output.view(m, local_ids.shape[1], k), local_ids
