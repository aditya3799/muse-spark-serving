import triton
import triton.language as tl

@triton.jit
def _fused_norm_rope_kernel_stacked(
    kv_ptr,  # [total_ctx, n_layers, kv_size * 2]
    k_norm_weight_ptr,  # [n_layers, head_dim]
    eps_ptr,  # [n_layers]
    cos_sin_cache_ptr,  # [max_pos, rotary_dim]
    positions_ptr,  # [total_ctx]
    k_out_ptr,  # [n_layers, total_ctx, num_kv_heads, head_dim]
    v_out_ptr,  # [n_layers, total_ctx, num_kv_heads, head_dim]
    kv_stride_ctx,
    kv_stride_layer,
    k_norm_weight_stride_layer,
    cos_sin_stride_pos,
    k_out_stride_layer,
    k_out_stride_ctx,
    k_out_stride_head,
    v_out_stride_layer,
    v_out_stride_ctx,
    v_out_stride_head,
    total_ctx,
    n_layers: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    kv_size: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary_dim: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    loc_ptr=None,
    commit_ptr=None,
    WIDTH: tl.constexpr = 0,
):
    """Fused RMSNorm(K) + RoPE(K) materialization. Grid: (total_ctx, num_kv_heads, n_layers)."""
    ctx_id = tl.program_id(0)
    head_id = tl.program_id(1)
    layer_id = tl.program_id(2)
    if ctx_id >= total_ctx or layer_id >= n_layers:
        return

    position = tl.load(positions_ptr + ctx_id)
    eps = tl.load(eps_ptr + layer_id).to(tl.float32)
    kv_base = kv_ptr + ctx_id * kv_stride_ctx + layer_id * kv_stride_layer
    k_base = kv_base + head_id * head_dim
    v_base = kv_base + kv_size + head_id * head_dim
    if WIDTH:
        if ctx_id % WIDTH >= tl.load(commit_ptr + ctx_id // WIDTH):
            return
        physical = tl.load(loc_ptr + ctx_id)
        key_storage = tl.load(k_out_ptr + layer_id).to(tl.pointer_type(kv_ptr.dtype.element_ty))
        value_storage = tl.load(v_out_ptr + layer_id).to(tl.pointer_type(kv_ptr.dtype.element_ty))
        k_write = key_storage + physical * k_out_stride_ctx + head_id * k_out_stride_head
        v_write = value_storage + physical * v_out_stride_ctx + head_id * v_out_stride_head
    else:
        k_write = (
            k_out_ptr
            + layer_id * k_out_stride_layer
            + ctx_id * k_out_stride_ctx
            + head_id * k_out_stride_head
        )
        v_write = (
            v_out_ptr
            + layer_id * v_out_stride_layer
            + ctx_id * v_out_stride_ctx
            + head_id * v_out_stride_head
        )

    offs = tl.arange(0, BLOCK_HD)
    mask_hd = offs < head_dim
    mask_half = offs < half_rotary_dim

    k_raw = tl.load(k_base + offs, mask=mask_hd, other=0.0).to(tl.float32)
    v_raw = tl.load(v_base + offs, mask=mask_hd, other=0.0)

    inv_rms = tl.rsqrt(tl.sum(k_raw * k_raw) / head_dim + eps)
    norm_w = tl.load(
        k_norm_weight_ptr + layer_id * k_norm_weight_stride_layer + offs,
        mask=mask_hd,
        other=1.0,
    ).to(tl.float32)
    k_normed = k_raw * inv_rms * norm_w

    cos_sin_base = cos_sin_cache_ptr + position * cos_sin_stride_pos
    cos_v = tl.load(cos_sin_base + offs, mask=mask_half, other=1.0).to(tl.float32)
    sin_v = tl.load(
        cos_sin_base + half_rotary_dim + offs, mask=mask_half, other=0.0
    ).to(tl.float32)

    k_first = tl.where(mask_half, k_normed, 0.0)
    k_second_raw = tl.load(
        k_base + half_rotary_dim + offs, mask=mask_half, other=0.0
    ).to(tl.float32)
    norm_w_second = tl.load(
        k_norm_weight_ptr
        + layer_id * k_norm_weight_stride_layer
        + half_rotary_dim
        + offs,
        mask=mask_half,
        other=1.0,
    ).to(tl.float32)
    k_second = k_second_raw * inv_rms * norm_w_second

    k_rot_first = k_first * cos_v - k_second * sin_v
    k_rot_second = k_second * cos_v + k_first * sin_v

    tl.store(v_write + offs, v_raw, mask=mask_hd)
    tl.store(k_write + offs, k_rot_first.to(v_raw.dtype), mask=mask_half)
    tl.store(
        k_write + half_rotary_dim + offs, k_rot_second.to(v_raw.dtype), mask=mask_half
    )
    mask_pass = (offs >= rotary_dim) & (offs < head_dim)
    tl.store(k_write + offs, k_normed.to(v_raw.dtype), mask=mask_pass)
