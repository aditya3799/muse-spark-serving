# okc: problem=muse_moe_w4a16
from typing import Optional

import torch
import triton
import triton.language as tl

_MIN_TOKENS_FOR_FUSED = 32


@triton.jit
def _post_expert_norm_weighted_sum_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    out_ptr,
    ids_ptr,
    scale_ptr,
    HAS_SCALE: tl.constexpr,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    eps,
    APPLY_NORM: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    WEIGHT_FIRST: tl.constexpr,
    HAS_IDS: tl.constexpr,
    ROUND_PRODUCT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    tok = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    if HAS_WEIGHT:
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for k in tl.static_range(TOPK):
        base = (tok * TOPK + k) * H
        live = True
        expert_id = -1
        if HAS_IDS:
            expert_id = tl.load(ids_ptr + tok * TOPK + k)
            live = expert_id >= 0
        if live:
            x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            if HAS_SCALE:
                scale = tl.load(scale_ptr + expert_id).to(tl.float32)
                # Preserve BF16 multiplication before FP32 norm/reduction.
                x = (x * scale).to(tl.bfloat16).to(tl.float32)

            if APPLY_NORM:
                if HAS_WEIGHT and WEIGHT_FIRST:
                    y = x * w
                    var = tl.sum(y * y, axis=0) / H
                    y = y * tl.rsqrt(var + eps)
                elif HAS_WEIGHT:
                    var = tl.sum(x * x, axis=0) / H
                    y = x * tl.rsqrt(var + eps) * w
                else:
                    var = tl.sum(x * x, axis=0) / H
                    y = x * tl.rsqrt(var + eps)
            else:
                y = x

            s = tl.load(s_ptr + tok * TOPK + k).to(tl.float32)
            product = y * s
            if ROUND_PRODUCT:
                product = product.to(x_ptr.dtype.element_ty).to(tl.float32)
            acc += product

    tl.store(out_ptr + tok * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


# Two-phase work decomposition inspired by muse-okc-submissions' MoE bs1
# redesign (github.com/gauravjain14/muse-okc-submissions) and our standalone
# decode work: parallelize independent expert
# reductions, then distribute the combine over output tiles. No weight repack
# or change to the precision / ordering of the per-slot weighted sum is needed.
@triton.jit
def _post_expert_inv_rms_kernel(
    x_ptr,
    w_ptr,
    ids_ptr,
    inv_ptr,
    scale_ptr,
    HAS_SCALE: tl.constexpr,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    eps,
    HAS_WEIGHT: tl.constexpr,
    WEIGHT_FIRST: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    slot = tl.program_id(0)
    expert_id = tl.load(ids_ptr + slot)
    live = expert_id >= 0
    inv = 0.0
    if live:
        offs = tl.arange(0, BLOCK_H)
        x = tl.load(x_ptr + slot * H + offs, mask=offs < H, other=0.0).to(tl.float32)
        if HAS_SCALE:
            scale = tl.load(scale_ptr + expert_id).to(tl.float32)
            # Preserve BF16 multiplication before FP32 norm/reduction.
            x = (x * scale).to(tl.bfloat16).to(tl.float32)
        if HAS_WEIGHT and WEIGHT_FIRST:
            w = tl.load(w_ptr + offs, mask=offs < H, other=0.0).to(tl.float32)
            x = x * w
        var = tl.sum(x * x, axis=0) / H
        inv = tl.rsqrt(var + eps)
    tl.store(inv_ptr + slot, inv)


@triton.jit
def _post_expert_tiled_combine_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    ids_ptr,
    inv_ptr,
    out_ptr,
    scale_ptr,
    HAS_SCALE: tl.constexpr,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    WEIGHT_FIRST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tok = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    if HAS_WEIGHT:
        w = tl.load(w_ptr + offs, mask=offs < H, other=0.0).to(tl.float32)
    acc = tl.zeros([BLOCK], tl.float32)
    for k in tl.static_range(TOPK):
        slot = tok * TOPK + k
        expert_id = tl.load(ids_ptr + slot)
        live = expert_id >= 0
        if live:
            x = tl.load(x_ptr + slot * H + offs, mask=offs < H, other=0.0).to(
                tl.float32
            )
            if HAS_SCALE:
                scale = tl.load(scale_ptr + expert_id).to(tl.float32)
                # Preserve BF16 multiplication before FP32 norm/reduction.
                x = (x * scale).to(tl.bfloat16).to(tl.float32)
            inv = tl.load(inv_ptr + slot)
            if HAS_WEIGHT and WEIGHT_FIRST:
                y = (x * w) * inv
            elif HAS_WEIGHT:
                y = (x * inv) * w
            else:
                y = x * inv
            score = tl.load(s_ptr + slot).to(tl.float32)
            acc += y * score
    tl.store(out_ptr + tok * H + offs, acc.to(out_ptr.dtype.element_ty), mask=offs < H)


def fused_post_expert_norm_weighted_sum(
    expert_output: torch.Tensor,
    topk_scores: torch.Tensor,
    norm_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    weight_first: bool = True,
    apply_norm: bool = True,
    slot_expert_ids: Optional[torch.Tensor] = None,
    round_product: bool = False,
    expert_output_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalize and combine experts without reading inactive slots.

    round_product preserves input-dtype multiplication before FP32 accumulation
    when combining an already normalized low-precision tensor.
    """
    x = expert_output.contiguous()
    tokens, topk, hidden = x.shape
    if expert_output_scale is not None:
        if slot_expert_ids is None:
            raise ValueError("Expert output scales require local slot IDs")
        if x.dtype != torch.bfloat16 or expert_output_scale.dtype != torch.bfloat16:
            raise TypeError("Fused expert output scaling requires BF16 tensors")
        if expert_output_scale.ndim != 1 or expert_output_scale.numel() == 0:
            raise ValueError("Expected a nonempty scale vector")
        if not expert_output_scale.is_contiguous():
            raise ValueError("Expert output scales must be contiguous")
        if slot_expert_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("Local expert IDs must be integer tensors")
        if slot_expert_ids.numel() != tokens * topk:
            raise ValueError("Expected one local expert ID per output slot")
        if not x.is_cuda or not x.device == slot_expert_ids.device == expert_output_scale.device:
            raise ValueError("Outputs, local IDs and scales must share one CUDA device")
    out = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    if tokens == 0:
        return out
    if tokens < _MIN_TOKENS_FOR_FUSED and slot_expert_ids is None and not round_product:
        y = x
        if apply_norm:
            f = y.float()
            if norm_weight is not None and weight_first:
                f = f * norm_weight.float()
                f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
            elif norm_weight is not None:
                f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
                f = f * norm_weight.float()
            else:
                f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
            y = f
        else:
            y = y.float()
        y = y * topk_scores.float().unsqueeze(-1)
        return y.sum(dim=1).to(x.dtype)

    block_h = triton.next_power_of_2(hidden)
    # The one-CTA-per-token kernel serializes TOPK full-width reductions and
    # leaves most SMs idle for small routed batches. Keep its path for larger
    # batches and uncommon arithmetic modes, where an extra launch may lose.
    if (
        x.is_cuda
        and 2 <= tokens <= 8
        and 2 <= topk <= 32
        and 1024 <= hidden <= 16384
        and slot_expert_ids is not None
        and apply_norm
        and not round_product
    ):
        ids = slot_expert_ids.contiguous()
        weight = norm_weight.contiguous() if norm_weight is not None else x
        inv = torch.empty((tokens * topk,), device=x.device, dtype=torch.float32)
        _post_expert_inv_rms_kernel[(tokens * topk,)](
            x,
            weight,
            ids,
            inv,
            expert_output_scale if expert_output_scale is not None else x,
            HAS_SCALE=expert_output_scale is not None,
            TOPK=topk,
            H=hidden,
            eps=eps,
            HAS_WEIGHT=norm_weight is not None,
            WEIGHT_FIRST=weight_first,
            BLOCK_H=block_h,
            num_warps=8 if block_h >= 4096 else 4,
            num_stages=2,
        )
        _post_expert_tiled_combine_kernel[(tokens, triton.cdiv(hidden, 256))](
            x,
            weight,
            topk_scores.contiguous(),
            ids,
            inv,
            out,
            expert_output_scale if expert_output_scale is not None else x,
            HAS_SCALE=expert_output_scale is not None,
            TOPK=topk,
            H=hidden,
            HAS_WEIGHT=norm_weight is not None,
            WEIGHT_FIRST=weight_first,
            BLOCK=256,
            num_warps=4,
        )
        return out
    _post_expert_norm_weighted_sum_kernel[(tokens,)](
        x,
        norm_weight.contiguous() if norm_weight is not None else x,
        topk_scores.contiguous(),
        out,
        slot_expert_ids.contiguous() if slot_expert_ids is not None else x,
        expert_output_scale if expert_output_scale is not None else x,
        HAS_SCALE=expert_output_scale is not None,
        TOPK=topk,
        H=hidden,
        eps=eps,
        APPLY_NORM=apply_norm,
        HAS_WEIGHT=norm_weight is not None,
        WEIGHT_FIRST=weight_first,
        HAS_IDS=slot_expert_ids is not None,
        ROUND_PRODUCT=round_product,
        BLOCK_H=block_h,
        num_warps=8 if block_h >= 4096 else 4,
        num_stages=2,
    )
    return out


__all__ = ["fused_post_expert_norm_weighted_sum"]

def extras(weights):
    scale = weights["w2_gs"][:32]
    norm = (weights["post_norm_w"].float() + 1).bfloat16()
    lookup = torch.full((257,), -1, device=norm.device, dtype=torch.int32)
    lookup[1:33] = torch.arange(32, device=norm.device, dtype=torch.int32)
    return norm, scale.bfloat16().contiguous(), lookup

def kernel_fn(data, state):
    cfg = data["config"]
    if (cfg["local_start"], cfg["local_count"]) != (0, 32):
        raise ValueError("expected the EP8 local expert bank")
    bank, norm, scale, lookup = state
    ids = lookup[data["topk_ids"].clamp_min(-1).to(torch.int32) + 1].contiguous()
    x = data["hidden_states"].contiguous()
    y = project(bank,x,ids)
    return fused_post_expert_norm_weighted_sum(y,data["topk_weights"],norm_weight=norm,
        weight_first=True,slot_expert_ids=ids,expert_output_scale=scale)

"""Exact BF16 global scaling, skipping non-local expert routes."""
import torch
import triton
import triton.language as tl


@triton.jit
def _scale_rows_kernel(x_ptr, ids_ptr, scale_ptr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    expert = tl.load(ids_ptr + row)
    if expert >= 0:
        offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(x_ptr + row * WIDTH + offset, mask=offset < WIDTH, other=0).to(tl.float32)
        scale = tl.load(scale_ptr + expert).to(tl.float32)
        # Preserve the separate BF16 multiplication after the BF16 GEMM output.
        result = (value * scale).to(x_ptr.dtype.element_ty)
        tl.store(x_ptr + row * WIDTH + offset, result, mask=offset < WIDTH)


def scale_active_rows(output, local_ids, scales):
    if output.ndim != 2 or scales.ndim != 1:
        raise ValueError("Expected a matrix output and one scale per expert")
    if output.dtype != torch.bfloat16 or scales.dtype != torch.bfloat16:
        raise TypeError("Global scaling requires BF16 output and scales")
    if local_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("Local expert IDs must be integer tensors")
    if not output.is_cuda or not output.device == local_ids.device == scales.device:
        raise ValueError("Output, IDs and scales must share one CUDA device")
    if not all(t.is_contiguous() for t in (output, local_ids, scales)):
        raise ValueError("Output, IDs and scales must be contiguous")
    if output.shape[0] != local_ids.numel() or scales.numel() == 0:
        raise ValueError("Expected one local ID per output row and nonempty scales")
    width = output.shape[1]
    if width == 0:
        raise ValueError("Output width must be positive")
    if output.shape[0] == 0:
        return
    block = min(8192, triton.next_power_of_2(width))
    _scale_rows_kernel[(output.shape[0], triton.cdiv(width, block))](
        output, local_ids, scales, WIDTH=width, BLOCK=block,
        num_warps=8 if block >= 8192 else 4,
    )

"""Histogram-selected expert partitions sharing canonical outputs.

Unsupported inputs retain the original helper. Both up GEMMs must finish before any down GEMM because
cache13 aliases their outputs. This module has no host reads of route values.
"""
import torch
import triton
import triton.language as tl
from copy import deepcopy


def select_ep_config(intervals, actual_routes, local_experts, global_experts):
    tune_m = (actual_routes * local_experts + global_experts - 1) // global_experts
    config = next((deepcopy(c) for lo, hi, c in intervals if lo < tune_m <= hi), None)
    if config is None or config.get("use_stream_k") is not False or config.get("use_f16_accum") is not False:
        return None
    return {"tune_m": tune_m, "valid_shape_m": actual_routes, "config": config}



@triton.jit
def _partition_kernel(ids_ptr, light_ptr, heavy_ptr, count_ptr,
                      ROUTES: tl.constexpr, EXPERTS: tl.constexpr,
                      CUTOFF: tl.constexpr, BLOCK: tl.constexpr, BINS: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    ids = tl.load(ids_ptr + offsets, mask=offsets < ROUTES, other=-1)
    valid = (offsets < ROUTES) & (ids >= 0) & (ids < EXPERTS)
    buckets = tl.where(valid, ids, BINS - 1).to(tl.int32)
    counts = tl.histogram(buckets, BINS)
    safe_ids = tl.where(valid, ids, 0).to(tl.int32)
    route_counts = tl.gather(counts, safe_ids, 0)
    light = valid & (route_counts <= CUTOFF)
    heavy = valid & (route_counts > CUTOFF)
    tl.store(light_ptr + offsets, tl.where(light, ids, -1), mask=offsets < ROUTES)
    tl.store(heavy_ptr + offsets, tl.where(heavy, ids, -1), mask=offsets < ROUTES)
    expert_offsets = tl.arange(0, BINS)
    tl.store(count_ptr + expert_offsets, counts, mask=expert_offsets < EXPERTS)


def partition_routes(local_ids, experts, cutoff):
    routes = local_ids.numel()
    if not (0 < routes <= 4096 and 0 < experts <= 256 and cutoff > 0):
        raise ValueError('Partitioning supports 1..4096 routes and 1..256 experts')
    if not local_ids.is_cuda or not local_ids.is_contiguous() or local_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('Expected contiguous CUDA integer route IDs')
    light = torch.empty(local_ids.shape, dtype=torch.int32, device=local_ids.device)
    heavy = torch.empty_like(light)
    counts = torch.empty((experts,), dtype=torch.int32, device=local_ids.device)
    _partition_kernel[(1,)](local_ids, light, heavy, counts,
                           ROUTES=routes, EXPERTS=experts, CUTOFF=cutoff,
                           BLOCK=triton.next_power_of_2(routes),
                           BINS=triton.next_power_of_2(experts + 1),
                           num_warps=8 if routes > 1024 else 4)
    return light, heavy, counts


def choose_partition_configs(layer, actual_routes, global_experts):
    experts = layer.w13_post_scale.numel()
    default = {s:next(c for lo,hi,c in intervals if lo < actual_routes <= hi)
               for s,intervals in layer.fp32_configs.items()}
    selections = {s:select_ep_config(intervals,actual_routes,experts,global_experts)
                  for s,intervals in layer.fp32_configs.items()}
    if any(value is None for value in selections.values()):
        return None
    light = {stage:value['config'] for stage,value in selections.items()}
    if not all(c.get('use_f16_accum') is False and c.get('use_stream_k') is False
               for c in list(default.values()) + list(light.values())):
        return None
    # Only explore pairs where expected-local tuning uses no larger row tile.
    distinct = any(light[s] != default[s] for s in default)
    smaller = all(light[s]['block_shape'][0] <= default[s]['block_shape'][0] for s in default)
    if not distinct or not smaller:
        return None
    cutoff = min(c['block_shape'][0] for c in light.values())
    return {'light':light,'heavy':default,'cutoff':cutoff}


def split_expert_outputs(original_helper, layer, x, local_ids, torch, global_experts=None,
                         *, retain=False, defer_downscale=False):
    from humming.layer import HummingMethod
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.kernels.ops.activation.activation import silu_and_mul
    def fallback():
        return original_helper(layer,x,local_ids,torch,retain=retain,
                               defer_downscale=defer_downscale)
    if (x.ndim != 2 or local_ids.ndim != 2 or x.dtype != torch.bfloat16
            or local_ids.dtype not in (torch.int32,torch.int64)
            or not x.is_cuda or x.device != local_ids.device
            or not x.is_contiguous() or not local_ids.is_contiguous()
            or local_ids.shape[0] != x.shape[0]):
        return fallback()
    valid = local_ids.numel()
    experts = layer.w13_post_scale.numel()
    if (valid == 0 or valid > 4096 or not 0 < experts <= 256
            or not isinstance(global_experts,int) or isinstance(global_experts,bool)
            or global_experts < experts or global_experts % experts
            or layer.w2_post_scale.numel() != experts):
        return fallback()
    plan = choose_partition_configs(layer,valid,global_experts)
    if plan is None:
        return fallback()
    light_ids,heavy_ids,counts = partition_routes(local_ids,experts,plan['cutoff'])
    partitions = [('light',light_ids),('heavy',heavy_ids)]
    aligned = {}
    def alignment(name,ids,block):
        key = (name,block)
        if key not in aligned:
            aligned[key] = moe_align_block_size(ids,block,experts,ignore_invalid_expert=True)
        return aligned[key]
    m,k = x.shape
    topk = local_ids.shape[1]
    intermediate = layer.humming_metas['w2'].shape_k
    cache13 = torch.empty((m*topk*max(2*intermediate,k),),device=x.device,dtype=x.dtype)
    cache1 = cache13[:m*topk*2*intermediate].view(m*topk,2*intermediate)
    for name,ids in partitions:
        config = plan[name]['w13']
        sorted_ids,expert_ids,padded = alignment(name,ids,config['block_shape'][0])
        HummingMethod.forward_layer(layer,inputs=x,outputs=cache1,top_k=topk,sublayer_name='w13',
            tuning_config=config,sorted_ids=sorted_ids,expert_ids=expert_ids,num_tokens_padded=padded,
            valid_shape_m=valid,compute_config={'gemm_type':'indexed','use_f16_accum':False})
    # Both up partitions are complete before scaling/activation or down-buffer reuse.
    scale_active_rows(cache1,local_ids,layer.w13_post_scale)
    cache2 = torch.empty((m*topk,intermediate),device=x.device,dtype=x.dtype)
    silu_and_mul(cache1,cache2,expert_ids=local_ids.flatten(),expert_step=1)
    cache3 = cache13[:m*topk*k].view(m*topk,k)
    for name,ids in partitions:
        config = plan[name]['w2']
        sorted_ids,expert_ids,padded = alignment(name,ids,config['block_shape'][0])
        HummingMethod.forward_layer(layer,inputs=cache2,outputs=cache3,top_k=1,sublayer_name='w2',
            tuning_config=config,sorted_ids=sorted_ids,expert_ids=expert_ids,num_tokens_padded=padded,
            valid_shape_m=valid,compute_config={'gemm_type':'indexed','use_f16_accum':False})
    if not defer_downscale:
        scale_active_rows(cache3,local_ids,layer.w2_post_scale)
    return cache3.view(m,topk,k),local_ids


# Scalable large-route path; the existing small-route implementation is preserved.
_original_split_expert_outputs = split_expert_outputs

"""Isolated larger-prefill partition prototype; canonical weights/math unchanged.

Only the original4096-route cap and partition implementation differ. Existing
<=4096 routes dispatch the exact scored partition. No host reads of route data.
"""
from copy import deepcopy
import torch
import triton
import triton.language as tl
original_choose_partition_configs = choose_partition_configs

@triton.jit
def _partial_hist(ids,partials,ROUTES:tl.constexpr,EXPERTS:tl.constexpr,BINS:tl.constexpr,BLOCK:tl.constexpr):
    tile=tl.program_id(0)
    ix=tile*BLOCK+tl.arange(0,BLOCK)
    value=tl.load(ids+ix,mask=ix<ROUTES,other=-1)
    valid=(ix<ROUTES)&(value>=0)&(value<EXPERTS)
    bucket=tl.where(valid,value,BINS-1).to(tl.int32)
    histogram=tl.histogram(bucket,BINS)
    e=tl.arange(0,BINS)
    tl.store(partials+tile*EXPERTS+e,histogram,mask=e<EXPERTS)

@triton.jit
def _reduce_hist(partials,counts,EXPERTS:tl.constexpr,TILES:tl.constexpr,BLOCK:tl.constexpr):
    e=tl.program_id(0)
    ix=tl.arange(0,BLOCK)
    values=tl.load(partials+ix*EXPERTS+e,mask=ix<TILES,other=0)
    tl.store(counts+e,tl.sum(values,0))

@triton.jit
def _classify(ids,counts,light,heavy,ROUTES:tl.constexpr,EXPERTS:tl.constexpr,CUTOFF:tl.constexpr,BLOCK:tl.constexpr):
    ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    value=tl.load(ids+ix,mask=ix<ROUTES,other=-1)
    valid=(ix<ROUTES)&(value>=0)&(value<EXPERTS)
    count=tl.load(counts+tl.where(valid,value,0))
    tl.store(light+ix,tl.where(valid&(count<=CUTOFF),value,-1).to(tl.int32),mask=ix<ROUTES)
    tl.store(heavy+ix,tl.where(valid&(count>CUTOFF),value,-1).to(tl.int32),mask=ix<ROUTES)

def partition_routes_scalable(local_ids,experts,cutoff):
    routes=local_ids.numel()
    if not (0<routes<=131072 and 0<experts<=256 and cutoff>0):
        raise ValueError('Unsupported route histogram geometry')
    if not local_ids.is_cuda or not local_ids.is_contiguous() or local_ids.dtype not in (torch.int32,torch.int64):
        raise ValueError('Expected contiguous CUDA integer route IDs')
    if routes<=4096:return partition_routes(local_ids,experts,cutoff)
    block=2048;tiles=triton.cdiv(routes,block)
    partials=torch.empty((tiles,experts),dtype=torch.int32,device=local_ids.device)
    counts=torch.empty((experts,),dtype=torch.int32,device=local_ids.device)
    light=torch.empty(local_ids.shape,dtype=torch.int32,device=local_ids.device)
    heavy=torch.empty_like(light)
    _partial_hist[(tiles,)](local_ids,partials,ROUTES=routes,EXPERTS=experts,BINS=triton.next_power_of_2(experts+1),BLOCK=block,num_warps=4)
    _reduce_hist[(experts,)](partials,counts,EXPERTS=experts,TILES=tiles,BLOCK=triton.next_power_of_2(tiles),num_warps=4)
    _classify[(triton.cdiv(routes,256),)](local_ids,counts,light,heavy,ROUTES=routes,EXPERTS=experts,CUTOFF=cutoff,BLOCK=256,num_warps=4)
    return light,heavy,counts

def _split_large_prefill_expert_outputs(original_helper, layer, x, local_ids, torch, global_experts=None,
                         *, retain=False, defer_downscale=False):
    from humming.layer import HummingMethod
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.kernels.ops.activation.activation import silu_and_mul
    def fallback():
        return original_helper(layer,x,local_ids,torch,retain=retain,
                               defer_downscale=defer_downscale)
    if (x.ndim != 2 or local_ids.ndim != 2 or x.dtype != torch.bfloat16
            or local_ids.dtype not in (torch.int32,torch.int64)
            or not x.is_cuda or x.device != local_ids.device
            or not x.is_contiguous() or not local_ids.is_contiguous()
            or local_ids.shape[0] != x.shape[0]):
        return fallback()
    valid = local_ids.numel()
    experts = layer.w13_post_scale.numel()
    if (valid == 0 or valid > 131072 or not 0 < experts <= 256
            or not isinstance(global_experts,int) or isinstance(global_experts,bool)
            or global_experts < experts or global_experts % experts
            or layer.w2_post_scale.numel() != experts):
        return fallback()
    plan = _choose_large_prefill_configs(layer,valid,global_experts)
    if plan is None:
        return fallback()
    light_ids,heavy_ids,counts = partition_routes_scalable(local_ids,experts,plan['cutoff'])
    partitions = [('light',light_ids),('heavy',heavy_ids)]
    aligned = {}
    def alignment(name,ids,block):
        key = (name,block)
        if key not in aligned:
            aligned[key] = moe_align_block_size(ids,block,experts,ignore_invalid_expert=True)
        return aligned[key]
    m,k = x.shape
    topk = local_ids.shape[1]
    intermediate = layer.humming_metas['w2'].shape_k
    cache13 = torch.empty((m*topk*max(2*intermediate,k),),device=x.device,dtype=x.dtype)
    cache1 = cache13[:m*topk*2*intermediate].view(m*topk,2*intermediate)
    for name,ids in partitions:
        config = plan[name]['w13']
        sorted_ids,expert_ids,padded = alignment(name,ids,config['block_shape'][0])
        HummingMethod.forward_layer(layer,inputs=x,outputs=cache1,top_k=topk,sublayer_name='w13',
            tuning_config=config,sorted_ids=sorted_ids,expert_ids=expert_ids,num_tokens_padded=padded,
            valid_shape_m=valid,compute_config={'gemm_type':'indexed','use_f16_accum':False})
    # Both up partitions are complete before scaling/activation or down-buffer reuse.
    scale_active_rows(cache1,local_ids,layer.w13_post_scale)
    cache2 = torch.empty((m*topk,intermediate),device=x.device,dtype=x.dtype)
    silu_and_mul(cache1,cache2,expert_ids=local_ids.flatten(),expert_step=1)
    cache3 = cache13[:m*topk*k].view(m*topk,k)
    for name,ids in partitions:
        config = plan[name]['w2']
        sorted_ids,expert_ids,padded = alignment(name,ids,config['block_shape'][0])
        HummingMethod.forward_layer(layer,inputs=cache2,outputs=cache3,top_k=1,sublayer_name='w2',
            tuning_config=config,sorted_ids=sorted_ids,expert_ids=expert_ids,num_tokens_padded=padded,
            valid_shape_m=valid,compute_config={'gemm_type':'indexed','use_f16_accum':False})
    if not defer_downscale:
        scale_active_rows(cache3,local_ids,layer.w2_post_scale)
    return cache3.view(m,topk,k),local_ids

def _choose_large_prefill_configs(layer, actual_routes, global_experts):
    # The public wrapper dispatches these routes through the original function.
    # Retain the original selector semantics too if called directly.
    if type(actual_routes) is not int:
        return None
    if actual_routes <= 4096:
        return original_choose_partition_configs(layer, actual_routes, global_experts)
    if (type(actual_routes) is not int or actual_routes > 131072
            or type(global_experts) is not int or global_experts <= 0):
        return None
    try:
        admitted = getattr(layer, "_humming_large_prefill_configs", None)
        if type(admitted) is not frozenset or not admitted:
            return None
        intervals = layer.fp32_configs
        if type(intervals) is not dict or set(intervals) != {"w13", "w2"}:
            return None
        experts = layer.w13_post_scale.numel()
        if (type(experts) is not int or not 0 < experts <= 256
                or layer.w2_post_scale.numel() != experts
                or global_experts < experts or global_experts % experts):
            return None
        tune_m = (actual_routes * experts + global_experts - 1) // global_experts
        # Reject incomplete/ambiguous interval tables before the scored chooser.
        for stage in ("w13", "w2"):
            table = intervals[stage]
            if not isinstance(table, (list, tuple)) or not table:
                return None
            previous_hi = None
            for entry in table:
                if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                    return None
                lo, hi, config = entry
                if (type(lo) is not int or type(hi) is not int or lo < 0 or hi <= lo
                        or type(config) is not dict
                        or (previous_hi is not None and lo < previous_hi)):
                    return None
                previous_hi = hi
            if any(sum(lo < value <= hi for lo, hi, _ in table) != 1
                   for value in (actual_routes, tune_m)):
                return None
        plan = original_choose_partition_configs(layer, actual_routes, global_experts)
        if plan is None:
            return None
        plan = deepcopy(plan)
        for stage in ("w13", "w2"):
            light, heavy = plan["light"][stage], plan["heavy"][stage]
            for config in (light, heavy):
                if (config.get("use_f16_accum") is not False
                        or config.get("use_stream_k") is not False):
                    return None
                for key in ("block_shape", "warp_shape"):
                    shape = config.get(key)
                    if (not isinstance(shape, (list, tuple)) or len(shape) != 3
                            or any(type(v) is not int or v <= 0 for v in shape)):
                        return None
                if any(b % w for b, w in zip(config["block_shape"], config["warp_shape"])):
                    return None
            light["block_shape"] = list(light["block_shape"])
            light["warp_shape"] = list(light["warp_shape"])
            light["block_shape"][2] = heavy["block_shape"][2]
            light["warp_shape"][2] = heavy["warp_shape"][2]
            if light["block_shape"][0] > heavy["block_shape"][0]:
                return None
            for config in (light, heavy):
                block, warp = config["block_shape"], config["warp_shape"]
                if any(b % w for b, w in zip(block, warp)):
                    return None
                warps = 1
                for b, w in zip(block, warp):
                    warps *= b // w
                ws = config.get("use_warp_spec", False)
                if ws is not None and type(ws) is not bool:
                    return None
                if warps * 32 + (128 if ws is True else 0) > 1024:
                    return None
                signature = _large_prefill_config_signature(config)
                if (stage, signature) not in admitted:
                    return None
        # Original distinctness was checked before K repair. Recheck afterward.
        if all(plan["light"][stage] == plan["heavy"][stage] for stage in ("w13", "w2")):
            return None
        plan["cutoff"] = min(plan["light"][stage]["block_shape"][0] for stage in ("w13", "w2"))
        return plan
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, StopIteration, OverflowError):
        # Metadata-only rejection; no GPU work has begun and no launch is retried.
        return None



def split_expert_outputs(original_helper, layer, x, local_ids, torch, global_experts=None,
                         *, retain=False, defer_downscale=False):
    # Large-route admission is fixed during canonical weight preparation.
    if (getattr(layer, "_humming_large_prefill_partition_enabled", False) is not True
            or local_ids.numel() <= 4096):
        return _original_split_expert_outputs(
            original_helper, layer, x, local_ids, torch, global_experts,
            retain=retain, defer_downscale=defer_downscale)
    return _split_large_prefill_expert_outputs(
        original_helper, layer, x, local_ids, torch, global_experts,
        retain=retain, defer_downscale=defer_downscale)


def _large_prefill_config_signature(config):
    """Immutable full tuning signature; list/tuple shapes have equal semantics."""
    def freeze(value):
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("tuning keys must be strings")
            return ("dict", tuple((key, freeze(value[key])) for key in sorted(value)))
        if isinstance(value, (list, tuple)):
            return ("sequence", tuple(freeze(item) for item in value))
        if value is None:
            return ("none",)
        if type(value) in (bool, int, str):
            return (type(value).__name__, value)
        raise TypeError("unsupported tuning value")

    if not isinstance(config, dict):
        raise TypeError("tuning configuration must be a dictionary")
    return freeze(config)


def prepare_large_prefill_partition(layer, torch):
    """Admit finite native configurations before any forward/capture.

    This performs host configuration/device-property checks, never kernel build,
    module load or launch. It is not a native numerical or occupancy guarantee.
    The caller owns immutable metadata and graph invalidation after preparation.
    """
    layer._humming_large_prefill_partition_enabled = False
    layer._humming_large_prefill_configs = frozenset()
    try:
        from copy import deepcopy
        from dataclasses import fields

        from humming.config import ComputeConfig, LayerConfig, TuningConfig
        from humming.kernel.humming import HummingKernel
        from humming.utils.device import fits_device_smem

        metas = layer.humming_metas
        owned = layer.fp32_configs
        device = torch.device(layer.w13_post_scale.device)
        if device.type != "cuda" or layer.w2_post_scale.device != layer.w13_post_scale.device:
            return False
        if set(metas) != {"w13", "w2"} or not {"w13", "w2"}.issubset(owned):
            return False
        if (metas["w13"].shape_n != 2 * metas["w2"].shape_k
                or metas["w13"].shape_k != metas["w2"].shape_n
                or metas["w13"].num_experts != metas["w2"].num_experts):
            return False
        for meta in metas.values():
            if (str(meta.a_dtype) != "bfloat16" or str(meta.c_dtype) != "bfloat16"
                    or str(meta.b_dtype) != "float4e2m1" or str(meta.bs_dtype) != "float8e4m3"
                    or meta.input_scale_group_size != 0 or meta.weight_scale_group_size != 16
                    or meta.has_zero_point or meta.has_bias or meta.use_packed_k_layout):
                return False
        admitted = set()
        with torch.cuda.device(device):
            props = torch.cuda.get_device_properties(device)
            sm_version = props.major * 10 + props.minor

            class ValidationKernel(HummingKernel):
                # KernelRuntime normally caches by CUDA context and compiles in
                # init_kernel. Bypass both; retain the pinned native validators.
                def __new__(cls, *args, **kwargs):
                    return object.__new__(cls)

                def __post_init__(self):
                    LayerConfig.__post_init__(self)
                    ComputeConfig.__post_init__(self)
                    TuningConfig.__post_init__(self)
                    self.sm_version = sm_version
                    self.sm_version_str = str(sm_version) + ("a" if sm_version >= 90 else "")
                    self.check_shape()
                    self.check_dtype()
                    self.check_scale()
                    self.check_config()
                    self.mma_op_class = self.select_mma_op_class()

                def prepare(self):
                    raise RuntimeError("configuration admission cannot compile")

                def load_cubin(self):
                    raise RuntimeError("configuration admission cannot load a module")

            for stage in ("w13", "w2"):
                intervals = owned[stage]
                if not isinstance(intervals, (list, tuple)) or not 0 < len(intervals) <= 256:
                    return False
                metadata = {
                    field.name: getattr(metas[stage], field.name)
                    for field in fields(LayerConfig) if field.init
                }
                configs = {}
                for interval in intervals:
                    if not isinstance(interval, (list, tuple)) or len(interval) != 3:
                        return False
                    config = interval[2]
                    configs[_large_prefill_config_signature(config)] = deepcopy(config)
                checked = {}

                def legal(config):
                    signature = _large_prefill_config_signature(config)
                    if signature in checked:
                        return checked[signature]
                    checked[signature] = False
                    if len(checked) > 1024:
                        return False
                    if config.get("use_f16_accum") is not False or config.get("use_stream_k") is not False:
                        return False
                    for name in ("num_ctas_per_sm", "num_write_splits"):
                        count = config.get(name, 1)
                        if type(count) is not int or count < 1:
                            return False
                    for name in ("block_shape", "warp_shape"):
                        shape = config.get(name)
                        if not isinstance(shape, (list, tuple)) or len(shape) != 3:
                            return False
                        if any(type(value) is not int or value <= 0 for value in shape):
                            return False
                    # Keep layer format and compute contract owned by metadata.
                    tuning_names = {field.name for field in fields(TuningConfig) if field.init}
                    if set(config) - tuning_names - {"num_sms", "use_f16_accum"}:
                        return False
                    launch_sms = config.get("num_sms", 0)
                    if type(launch_sms) is not int or launch_sms < 0:
                        return False
                    values = dict(metadata)
                    values.update(gemm_type="indexed", use_f16_accum=False)
                    values.update(deepcopy(config))
                    values.pop("num_sms", None)
                    try:
                        validator = ValidationKernel(**values)
                        max_threads = getattr(props, "max_threads_per_block", 1024)
                        valid = (
                            0 < validator.num_threads <= max_threads
                            and validator.use_f16_accum is False
                            and validator.use_stream_k is False
                            and fits_device_smem(validator, validator, validator, device)
                        )
                    except Exception:
                        return False
                    checked[signature] = bool(valid)
                    return bool(valid)

                # Validate all unchanged heavy configurations first. Only their
                # K topologies may produce repaired light candidates.
                heavy_k = set()
                stage_admitted = set()
                for signature, config in configs.items():
                    if legal(config):
                        stage_admitted.add((stage, signature))
                        heavy_k.add((config["block_shape"][2], config["warp_shape"][2]))
                if not heavy_k or len(heavy_k) > 16:
                    return False
                eligible_pair = False
                for config in configs.values():
                    for block_k, warp_k in heavy_k:
                        repaired = deepcopy(config)
                        repaired["block_shape"] = list(repaired["block_shape"])
                        repaired["warp_shape"] = list(repaired["warp_shape"])
                        repaired["block_shape"][2] = block_k
                        repaired["warp_shape"][2] = warp_k
                        if legal(repaired):
                            stage_admitted.add((stage, _large_prefill_config_signature(repaired)))
                            eligible_pair = True
                if not eligible_pair:
                    return False
                admitted.update(stage_admitted)
        layer._humming_large_prefill_configs = frozenset(admitted)
        layer._humming_large_prefill_partition_enabled = True
        return True
    except Exception:
        # Missing or changed installed Humming APIs retain the scored helper.
        return False

def _prepare_humming_experts(raw, torch):
    """Pack one canonical NVFP4 layout with FP8 block scales.

The caller validates aligned shapes, allowing repacking without redundant
zero-padding copies. Global scales are applied after each BF16 GEMM output
to preserve the stock Marlin rounding boundary. Tuning overrides are owned
by this layer; Humming's cached defaults remain unchanged."""
    from humming import dtypes
    from humming.config import WeightScale2Type, WeightScaleType
    from humming.layer import HummingMethod
    from humming.schema import HummingInputSchema, HummingWeightSchema
    layer = torch.nn.Module()
    for prefix in ('w13', 'w2'):
        w = getattr(raw, prefix + '_weight')
        sf = getattr(raw, prefix + '_weight_scale')
        gs = getattr(raw, prefix + '_weight_scale_2')
        e, n, k2 = w.shape
        if w.dtype != torch.uint8 or sf.dtype != torch.float8_e4m3fn:
            raise ValueError('Humming NVFP4 requires uint8 codes and FP8 scales')
        if n % 64 or k2 * 2 % 64 or sf.shape != (e, n, k2 * 2 // 16):
            raise ValueError('Humming NVFP4 requires aligned shapes and group size 16')
        if gs.ndim != 1 or gs.shape[0] != e:
            raise ValueError('Humming NVFP4 requires one global scale per expert')
        layer.register_parameter(prefix + '_weight', torch.nn.Parameter(w.contiguous().view(torch.int32), requires_grad=False))
        layer.register_parameter(prefix + '_weight_scale', torch.nn.Parameter(sf.contiguous(), requires_grad=False))
        layer.register_buffer(prefix + '_post_scale', gs.to(torch.bfloat16).contiguous())
        schema = HummingWeightSchema(b_dtype=dtypes.float4e2m1, bs_dtype=dtypes.float8e4m3, weight_scale_group_size=16, weight_scale_type=WeightScaleType.GROUP, weight_scale_2_type=WeightScale2Type.NONE)
        meta = HummingMethod.prepare_layer_meta(layer, shape_n=n, shape_k=k2 * 2, weight_schema=schema, input_schema=HummingInputSchema(a_dtype=dtypes.bfloat16), num_experts=e, torch_dtype=torch.bfloat16, sublayer_name=prefix)
        assert meta.a_dtype == dtypes.bfloat16 and meta.c_dtype == dtypes.bfloat16
        assert not meta.use_int_weight_scale and (not meta.has_input_scale)
        HummingMethod.transform_humming_layer(layer, sublayer_name=prefix, already_padded=True)
    layer.register_buffer('locks', torch.zeros(max(2048, torch.cuda.get_device_properties(0).multi_processor_count * 8), device='cuda', dtype=torch.int32))
    import copy
    layer.fp32_configs = {}
    for stage in ('w13', 'w2'):
        default = HummingMethod.get_default_tuning_configs(layer, use_f16_accum=False, gemm_type='indexed', sublayer_name=stage)
        snapshot = copy.deepcopy(default)
        owned = copy.deepcopy(default)
        for lo, hi, c in owned:
            # Stream-K reduces partial outputs in BF16 even with FP32 MMA.
            # Disable it to preserve the stock FP32 reduction contract.
            c['use_stream_k'] = False
            c['use_f16_accum'] = False
        assert default == snapshot, 'Default tuning cache must stay unchanged'
        layer.fp32_configs[stage] = owned
    prepare_large_prefill_partition(layer, torch)
    return layer

def _humming_expert_outputs_default(layer, x, local_ids, torch, retain=False, *, defer_downscale=False):
    """Compute local routes with FP32 accumulation and no Stream-K reduction.

First and final outputs alias one buffer as in the stock Marlin path.
The caller applies the original per-expert norm and weighted finalization."""
    from humming.layer import HummingMethod
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.kernels.ops.activation.activation import silu_and_mul
    configs = layer.fp32_configs
    valid = local_ids.numel()
    selected = {s: next((c for lo, hi, c in cfg if lo < valid <= hi)) for s, cfg in configs.items()}
    assert all((c['use_stream_k'] is False and c['use_f16_accum'] is False for c in selected.values()))
    block = selected['w13']['block_shape'][0]
    sorted_ids, expert_ids, padded = moe_align_block_size(local_ids, block, layer.w13_post_scale.numel(), ignore_invalid_expert=True)
    m, k = x.shape
    topk = local_ids.shape[1]
    i = layer.humming_metas['w2'].shape_k
    cache13 = torch.empty((m * topk * max(2 * i, k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[:m * topk * 2 * i].view(m * topk, 2 * i)
    common = dict(sorted_ids=sorted_ids, expert_ids=expert_ids, num_tokens_padded=padded, valid_shape_m=valid, compute_config={'gemm_type': 'indexed', 'use_f16_accum': False})
    HummingMethod.forward_layer(layer, inputs=x, outputs=cache1, top_k=topk, sublayer_name='w13', tuning_config=selected['w13'], **common)
    # GEMM output has already rounded to BF16. Apply the BF16 global scale
    # in a separate BF16 operation, preserving Marlin's second rounding.
    scale_active_rows(cache1, local_ids, layer.w13_post_scale)
    cache2 = torch.empty((m * topk, i), device=x.device, dtype=x.dtype)
    silu_and_mul(cache1, cache2, expert_ids=local_ids.flatten(), expert_step=1)
    cache3 = cache13[:m * topk * k].view(m * topk, k)
    if selected['w2']['block_shape'][0] != block:
        second = moe_align_block_size(local_ids, selected['w2']['block_shape'][0], layer.w2_post_scale.numel(), ignore_invalid_expert=True)
        common.update(sorted_ids=second[0], expert_ids=second[1], num_tokens_padded=second[2])
    HummingMethod.forward_layer(layer, inputs=cache2, outputs=cache3, top_k=1, sublayer_name='w2', tuning_config=selected['w2'], **common)
    # Preserve the same BF16-output -> BF16-scale boundary for GEMM2.
    if not defer_downscale:
        scale_active_rows(cache3, local_ids, layer.w2_post_scale)
    return (cache3.view(m, topk, k), local_ids)

def _humming_expert_outputs(layer, x, local_ids, torch, retain=False, *,
                            defer_downscale=False, global_experts=None):
    return split_expert_outputs(
        _humming_expert_outputs_default,layer,x,local_ids,torch,global_experts,
        retain=retain,defer_downscale=defer_downscale,
    )
def prepare(weights):
    from types import SimpleNamespace
    raw = SimpleNamespace()
    for stage in ("w13","w2"):
        setattr(raw,stage+"_weight",weights[stage][:32])
        setattr(raw,stage+"_weight_scale",weights[stage+"_sf"][:32])
        setattr(raw,stage+"_weight_scale_2",weights[stage+"_gs"][:32])
    return (_prepare_humming_experts(raw,torch),*extras(weights))

def project(bank,x,ids):
    return _humming_expert_outputs(bank,x,ids,torch,defer_downscale=True,global_experts=256)[0]

