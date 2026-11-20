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
