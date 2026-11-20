def try_grouped(owner,x,topk_ids,scores):
    """Return the grouped output for a qualified graph shape, otherwise defer."""
    from sglang.kernels.ops.moe import muse_spark_humming_split as split
    if (x.device.type!="cuda" or x.dtype!=torch.bfloat16 or x.ndim!=2 or
        x.shape[0] not in (128,768) or x.shape[1]!=4096 or owner.global_num_experts!=256 or
        topk_ids.shape!=(x.shape[0],8) or scores.shape!=topk_ids.shape or
        scores.dtype!=torch.float32 or scores.device!=x.device or topk_ids.device!=x.device or
        topk_ids.dtype not in (torch.int32,torch.int64) or
        not owner._post_expert_norm_weight_first or owner._post_expert_norm_eps!=1e-8):
        return None
    norm=owner._post_expert_norm_weight
    if norm is None or norm.shape!=(4096,) or norm.device!=x.device or norm.dtype not in (torch.bfloat16,torch.float32):
        return None
    layer=owner._humming_layer
    if owner._expert_map is None or layer.w13_post_scale.numel()!=32 or layer.w2_post_scale.numel()!=32:
        return None
    sm=torch.cuda.get_device_capability(x.device)
    if sm not in ((8,6),(9,0)):
        return None
    plan=split._choose_large_prefill_configs(layer,topk_ids.numel(),256)
    if plan is None:
        return None
    for kind in ("light","heavy"):
        for stage in ("w13","w2"):
            cfg=plan[kind][stage]
            if (set(cfg)-{"block_shape","warp_shape","num_stages","use_f16_accum","use_stream_k","num_sms","num_ctas_per_sm","num_write_splits"} or
                    cfg.get("num_ctas_per_sm",1)!=1 or cfg.get("num_write_splits",1)!=1 or
                    cfg.get("use_f16_accum") is not False or cfg.get("use_stream_k") is not False):
                return None
            key=(sm,stage,tuple(cfg["block_shape"]),tuple(cfg["warp_shape"]),cfg["num_stages"])
            if key not in PROVEN_GEOMETRY:
                return None
    local=topk_ids.to(torch.int32)
    local=owner._expert_lookup[local.clamp_min(-1)+1].contiguous()
    return compute(layer,x.contiguous(),local,split,"split_fused",scores.contiguous(),norm.contiguous())
