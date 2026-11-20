"""Same-input one-layer diagnostic ONLY. Classify full IDs before masking.

NOT a multilayer solution: unused-tail hidden/router IDs change in later layers,
so that full-ID histogram need not equal baseline's histogram. No integration.
"""
import torch
from sglang.kernels.ops.moe.muse_spark_humming_split import choose_partition_configs,partition_routes
def masked_expert_outputs(original_helper, layer, x, local_ids, keep_mask, torch, global_experts=None,
                         *, retain=False, defer_downscale=False):
    from humming.layer import HummingMethod
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.muse_spark_humming_scale import scale_active_rows
    if keep_mask.ndim!=1 or keep_mask.shape[0]!=x.shape[0] or keep_mask.dtype!=torch.bool or keep_mask.device!=x.device:
        raise ValueError('Expected CUDA per-token bool keep mask')
    full_local_ids=local_ids
    local_ids=torch.where(keep_mask[:,None],full_local_ids,-1)
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
    light_ids,heavy_ids,counts = partition_routes(full_local_ids,experts,plan['cutoff'])
    light_ids=torch.where(keep_mask[:,None],light_ids,-1)
    heavy_ids=torch.where(keep_mask[:,None],heavy_ids,-1)
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
