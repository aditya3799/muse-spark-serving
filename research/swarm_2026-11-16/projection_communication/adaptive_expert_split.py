"""Isolated experiment: histogram-selected expert partitions sharing canonical outputs.

No frozen-source edits. Both up GEMMs must finish before any down GEMM because
cache13 aliases their outputs. This module has no host reads of route values.
"""
import torch
import triton
import triton.language as tl
from ep_tuning_selector import select_ep_config


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
        raise ValueError('Partition experiment supports1..4096 routes and1..256 experts')
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
    light = {s:select_ep_config(intervals,actual_routes,experts,global_experts)['config']
             for s,intervals in layer.fp32_configs.items()}
    assert all(c.get('use_f16_accum') is False and c.get('use_stream_k') is False
               for c in list(default.values()) + list(light.values()))
    # Only explore pairs where expected-local tuning uses no larger row tile.
    distinct = any(light[s] != default[s] for s in default)
    smaller = all(light[s]['block_shape'][0] <= default[s]['block_shape'][0] for s in default)
    if not distinct or not smaller:
        return None
    cutoff = min(c['block_shape'][0] for c in light.values())
    return {'light':light,'heavy':default,'cutoff':cutoff}


def split_expert_outputs(original_helper, layer, x, local_ids, torch, global_experts):
    from humming.layer import HummingMethod
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.muse_spark_humming_scale import scale_active_rows
    valid = local_ids.numel()
    experts = layer.w13_post_scale.numel()
    if valid == 0 or valid > 4096 or experts > 256:
        return original_helper(layer,x,local_ids,torch)
    plan = choose_partition_configs(layer,valid,global_experts)
    if plan is None:
        return original_helper(layer,x,local_ids,torch)
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
    scale_active_rows(cache3,local_ids,layer.w2_post_scale)
    return cache3.view(m,topk,k),local_ids
