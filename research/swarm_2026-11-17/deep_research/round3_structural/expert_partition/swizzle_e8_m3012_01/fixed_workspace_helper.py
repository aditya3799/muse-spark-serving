def _humming_expert_outputs_default(layer, x, local_ids, torch, retain=False, *, defer_downscale=False):
    """Compute local routes with FP32 accumulation and no Stream-K reduction.

First and final outputs alias one buffer as in the stock Marlin path.
The caller applies the original per-expert norm and weighted finalization."""
    from humming.layer import HummingMethod
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.muse_spark_humming_scale import scale_active_rows
    configs = layer.fp32_configs
    valid = local_ids.numel()
    selected = {s: next((c for lo, hi, c in cfg if lo < valid <= hi)) for s, cfg in configs.items()}
    assert all((c['use_stream_k'] is False and c['use_f16_accum'] is False for c in selected.values()))
    block = selected['w13']['block_shape'][0]
    sorted_ids, expert_ids, padded = moe_align_block_size(local_ids, block, layer.w13_post_scale.numel(), ignore_invalid_expert=True)
    m, k = x.shape
    topk = local_ids.shape[1]
    i = layer.humming_metas['w2'].shape_k
    cache13 = fixed_cache13
    cache1 = cache13[:m * topk * 2 * i].view(m * topk, 2 * i)
    common = dict(sorted_ids=sorted_ids, expert_ids=expert_ids, num_tokens_padded=padded, valid_shape_m=valid, compute_config={'gemm_type': 'indexed', 'use_f16_accum': False})
    HummingMethod.forward_layer(layer, inputs=x, outputs=cache1, top_k=topk, sublayer_name='w13', tuning_config=selected['w13'], **common)
    scale_active_rows(cache1, local_ids, layer.w13_post_scale)
    cache2 = fixed_cache2
    silu_and_mul(cache1, cache2, expert_ids=local_ids.flatten(), expert_step=1)
    cache3 = cache13[:m * topk * k].view(m * topk, k)
    if selected['w2']['block_shape'][0] != block:
        second = moe_align_block_size(local_ids, selected['w2']['block_shape'][0], layer.w2_post_scale.numel(), ignore_invalid_expert=True)
        common.update(sorted_ids=second[0], expert_ids=second[1], num_tokens_padded=second[2])
    HummingMethod.forward_layer(layer, inputs=cache2, outputs=cache3, top_k=1, sublayer_name='w2', tuning_config=selected['w2'], **common)
    if not defer_downscale:
        scale_active_rows(cache3, local_ids, layer.w2_post_scale)
    return (cache3.view(m, topk, k), local_ids)
