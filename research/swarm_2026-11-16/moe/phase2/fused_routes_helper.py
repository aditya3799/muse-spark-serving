"""Isolated scored-helper derivative: only alignment production changes."""
from copy import deepcopy
from fused_partition_align import fused_partition_align

def select_ep_config(intervals, actual_routes, local_experts, global_experts):
    tune_m = (actual_routes * local_experts + global_experts - 1) // global_experts
    config = next((deepcopy(c) for lo, hi, c in intervals if lo < tune_m <= hi), None)
    if config is None or config.get("use_stream_k") is not False or config.get("use_f16_accum") is not False:
        return None
    return {"tune_m": tune_m, "valid_shape_m": actual_routes, "config": config}

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

def split_expert_outputs_fused_routes(original_helper, layer, x, local_ids, torch, global_experts=None,
                         *, retain=False, defer_downscale=False):
    from humming.layer import HummingMethod
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.muse_spark_humming_scale import scale_active_rows
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
    # Pinned indexed scheduler may round out to cluster tiles if multicastB>1.
    # Keep the scored fallback for every unsupported derived tuning contract.
    if any(config.get('multi_cast_size_b', 1) != 1
           for name in ('light', 'heavy') for config in plan[name].values()):
        return fallback()
    keys = tuple(dict.fromkeys((name, plan[name][stage]['block_shape'][0])
                               for name in ('light', 'heavy') for stage in ('w13', 'w2')))
    aligned, counts = fused_partition_align(local_ids, experts, plan['cutoff'], keys)
    m,k = x.shape
    topk = local_ids.shape[1]
    intermediate = layer.humming_metas['w2'].shape_k
    cache13 = torch.empty((m*topk*max(2*intermediate,k),),device=x.device,dtype=x.dtype)
    cache1 = cache13[:m*topk*2*intermediate].view(m*topk,2*intermediate)
    for name in ('light', 'heavy'):
        config = plan[name]['w13']
        sorted_ids,expert_ids,padded = aligned[(name, config['block_shape'][0])]
        HummingMethod.forward_layer(layer,inputs=x,outputs=cache1,top_k=topk,sublayer_name='w13',
            tuning_config=config,sorted_ids=sorted_ids,expert_ids=expert_ids,num_tokens_padded=padded,
            valid_shape_m=valid,compute_config={'gemm_type':'indexed','use_f16_accum':False})
    # Both up partitions are complete before scaling/activation or down-buffer reuse.
    scale_active_rows(cache1,local_ids,layer.w13_post_scale)
    cache2 = torch.empty((m*topk,intermediate),device=x.device,dtype=x.dtype)
    silu_and_mul(cache1,cache2,expert_ids=local_ids.flatten(),expert_step=1)
    cache3 = cache13[:m*topk*k].view(m*topk,k)
    for name in ('light', 'heavy'):
        config = plan[name]['w2']
        sorted_ids,expert_ids,padded = aligned[(name, config['block_shape'][0])]
        HummingMethod.forward_layer(layer,inputs=cache2,outputs=cache3,top_k=1,sublayer_name='w2',
            tuning_config=config,sorted_ids=sorted_ids,expert_ids=expert_ids,num_tokens_padded=padded,
            valid_shape_m=valid,compute_config={'gemm_type':'indexed','use_f16_accum':False})
    if not defer_downscale:
        scale_active_rows(cache3,local_ids,layer.w2_post_scale)
    return cache3.view(m,topk,k),local_ids
