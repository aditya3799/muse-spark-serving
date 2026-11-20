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
            c['use_stream_k'] = False
            c['use_f16_accum'] = False
        assert default == snapshot, 'Default tuning cache must stay unchanged'
        layer.fp32_configs[stage] = owned
    return layer

def _humming_expert_outputs(layer, x, local_ids, torch, retain=False, *, defer_downscale=False):
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
    cache13 = torch.empty((m * topk * max(2 * i, k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[:m * topk * 2 * i].view(m * topk, 2 * i)
    common = dict(sorted_ids=sorted_ids, expert_ids=expert_ids, num_tokens_padded=padded, valid_shape_m=valid, compute_config={'gemm_type': 'indexed', 'use_f16_accum': False})
    HummingMethod.forward_layer(layer, inputs=x, outputs=cache1, top_k=topk, sublayer_name='w13', tuning_config=selected['w13'], **common)
    scale_active_rows(cache1, local_ids, layer.w13_post_scale)
    cache2 = torch.empty((m * topk, i), device=x.device, dtype=x.dtype)
    silu_and_mul(cache1, cache2, expert_ids=local_ids.flatten(), expert_step=1)
    cache3 = cache13[:m * topk * k].view(m * topk, k)
    if selected['w2']['block_shape'][0] != block:
        second = moe_align_block_size(local_ids, selected['w2']['block_shape'][0], layer.w2_post_scale.numel(), ignore_invalid_expert=True)
        common.update(sorted_ids=second[0], expert_ids=second[1], num_tokens_padded=second[2])
    HummingMethod.forward_layer(layer, inputs=cache2, outputs=cache3, top_k=1, sublayer_name='w2', tuning_config=selected['w2'], **common)
    if not defer_downscale:
        scale_active_rows(cache3, local_ids, layer.w2_post_scale)
    return (cache3.view(m, topk, k), local_ids)