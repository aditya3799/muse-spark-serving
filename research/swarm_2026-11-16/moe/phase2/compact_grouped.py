"""Compact activation grouped-masked experiment; same canonical weights.

Duplicate-safe capacity bounds every route. Only scratch policy and packing are
experimental; no production or H200 runtime validation is claimed.
"""
from copy import deepcopy
import torch
import triton
import triton.language as tl


@triton.jit
def _make_map(ids, slots, inverse, counts, ROUTES: tl.constexpr,
              CAPACITY: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    r = tl.arange(0, BLOCK)
    values = tl.load(ids + r, mask=r < ROUTES, other=-1)
    belongs = (r < ROUTES) & (values == expert)
    count = tl.sum(belongs.to(tl.int32), 0)
    ordinal = tl.cumsum(belongs.to(tl.int32)) - 1
    packed = expert * CAPACITY + ordinal
    tl.store(counts + expert, count)
    tl.store(slots + packed, r, mask=belongs)
    tl.store(inverse + r, packed, mask=belongs)
    tl.store(slots + expert * CAPACITY + r, ROUTES,
             mask=(r >= count) & (r < CAPACITY))


@triton.jit
def _pack_rows(source, slots, counts, destination,
               CAPACITY: tl.constexpr, WIDTH: tl.constexpr,
               SOURCE_DIVISOR: tl.constexpr, GEMM_BLOCK: tl.constexpr,
               BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    expert = row // CAPACITY
    local_row = row % CAPACITY
    count = tl.load(counts + expert)
    if local_row < tl.cdiv(count, GEMM_BLOCK) * GEMM_BLOCK:
        slot = tl.load(slots + row)
        columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        value = tl.load(source + (slot // SOURCE_DIVISOR) * WIDTH + columns,
                        mask=(local_row < count) & (columns < WIDTH), other=0)
        # Only padding rows that this expert's GEMM tile can read are zeroed.
        tl.store(destination + row * WIDTH + columns, value, mask=columns < WIDTH)


@triton.jit
def _scatter_rows(grouped, inverse, ids, scales, output,
                  WIDTH: tl.constexpr, EXPERTS: tl.constexpr,
                  APPLY_SCALE: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    expert = tl.load(ids + row)
    if (expert >= 0) & (expert < EXPERTS):
        packed = tl.load(inverse + row)
        columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        value = tl.load(grouped + packed * WIDTH + columns,
                        mask=columns < WIDTH, other=0)
        if APPLY_SCALE:
            scale = tl.load(scales + expert).to(tl.float32)
            # GEMM rounded to BF16; preserve the second BF16 scale rounding.
            value = (value.to(tl.float32) * scale).to(tl.bfloat16)
        tl.store(output + row * WIDTH + columns, value, mask=columns < WIDTH)


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


def compact_grouped_outputs(scored_helper, layer, grouped_configs, x, local_ids,
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
    # Total workspace, including ordinary route buffers and conservative driver
    # reserve. A per-call bound does not prove whole-model graph-pool memory.
    workspace = (2 * packed_rows * (max(k, intermediate) + max(2 * intermediate, k))
                 + 2 * routes * (max(2 * intermediate, k) + intermediate)
                 + 4 * packed_rows + 4 * routes + 4 * experts + 8 * 1024 * 1024)
    if workspace > workspace_limit:
        return fallback()
    device = x.device
    slots = torch.empty((packed_rows,), dtype=torch.int32, device=device)
    inverse = torch.empty((routes,), dtype=torch.int32, device=device)
    counts = torch.empty((experts,), dtype=torch.int32, device=device)
    _make_map[(experts,)](local_ids, slots, inverse, counts, ROUTES=routes,
                         CAPACITY=capacity, BLOCK=triton.next_power_of_2(capacity),
                         num_warps=8 if capacity > 1024 else 4)
    packed_input = torch.empty((packed_rows * max(k, intermediate),), dtype=x.dtype, device=device)
    packed_x = packed_input[:packed_rows * k].view(packed_rows, k)
    grouped13 = torch.empty((packed_rows * max(2 * intermediate, k),), dtype=x.dtype, device=device)
    grouped1 = grouped13[:packed_rows * 2 * intermediate].view(packed_rows, 2 * intermediate)
    cache13 = torch.empty((routes * max(2 * intermediate, k),), dtype=x.dtype, device=device)
    cache1 = cache13[:routes * 2 * intermediate].view(routes, 2 * intermediate)
    cache2 = torch.empty((routes, intermediate), dtype=x.dtype, device=device)
    _pack_rows[(packed_rows, triton.cdiv(k, 512))](
        x, slots, counts, packed_x, CAPACITY=capacity, WIDTH=k,
        SOURCE_DIVISOR=local_ids.shape[1], GEMM_BLOCK=up_block, BLOCK_N=512)
    # valid_shape_m chooses a tuning interval only. The pinned native launcher
    # passes inputs.size(0)=E*C to the grouped scheduler, so its row stride is C.
    common = dict(expert_layout=counts, top_k=1, valid_shape_m=routes,
                  compute_config={'gemm_type': 'grouped_masked', 'use_f16_accum': False})
    HummingMethod.forward_layer(layer, inputs=packed_x, outputs=grouped1,
                                sublayer_name='w13', tuning_config=selected['w13'], **common)
    _scatter_rows[(routes, triton.cdiv(2 * intermediate, 512))](
        grouped1, inverse, local_ids, layer.w13_post_scale, cache1,
        WIDTH=2 * intermediate, EXPERTS=experts, APPLY_SCALE=True, BLOCK_N=512)
    silu_and_mul(cache1, cache2, expert_ids=local_ids.flatten(), expert_step=1)
    # Up GEMM and scatter are complete before either grouped buffer is reused.
    packed_activation = packed_input[:packed_rows * intermediate].view(packed_rows, intermediate)
    _pack_rows[(packed_rows, triton.cdiv(intermediate, 512))](
        cache2, slots, counts, packed_activation, CAPACITY=capacity, WIDTH=intermediate,
        SOURCE_DIVISOR=1, GEMM_BLOCK=down_block, BLOCK_N=512)
    grouped3 = grouped13[:packed_rows * k].view(packed_rows, k)
    HummingMethod.forward_layer(layer, inputs=packed_activation, outputs=grouped3,
                                sublayer_name='w2', tuning_config=selected['w2'], **common)
    cache3 = cache13[:routes * k].view(routes, k)
    _scatter_rows[(routes, triton.cdiv(k, 512))](
        grouped3, inverse, local_ids, layer.w2_post_scale, cache3,
        WIDTH=k, EXPERTS=experts, APPLY_SCALE=False, BLOCK_N=512)
    if not defer_downscale:
        scale_active_rows(cache3, local_ids, layer.w2_post_scale)
    return cache3.view(m, local_ids.shape[1], k), local_ids
