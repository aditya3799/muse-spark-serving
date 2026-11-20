"""Isolated larger-prefill partition prototype; canonical weights/math unchanged.

Only the original4096-route cap and partition implementation differ. Existing
<=4096 routes dispatch the exact scored partition. No host reads of route data.
"""
from copy import deepcopy
import torch
import triton
import triton.language as tl
from sglang.kernels.ops.moe.muse_spark_humming_split import choose_partition_configs as original_choose_partition_configs, partition_routes

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

def split_expert_outputs_scalable(original_helper, layer, x, local_ids, torch, global_experts=None,
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
    if (valid == 0 or valid > 131072 or not 0 < experts <= 256
            or not isinstance(global_experts,int) or isinstance(global_experts,bool)
            or global_experts < experts or global_experts % experts
            or layer.w2_post_scale.numel() != experts):
        return fallback()
    plan = choose_partition_configs(layer,valid,global_experts)
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

def choose_partition_configs(layer, actual_routes, global_experts):
    plan = original_choose_partition_configs(layer, actual_routes, global_experts)
    if plan is None or actual_routes <= 4096:
        return plan
    plan = deepcopy(plan)
    for stage in ('w13', 'w2'):
        light, heavy = plan['light'][stage], plan['heavy'][stage]
        light['block_shape'] = list(light['block_shape'])
        light['warp_shape'] = list(light['warp_shape'])
        light['block_shape'][2] = heavy['block_shape'][2]
        light['warp_shape'][2] = heavy['warp_shape'][2]
        assert light['use_f16_accum'] is False and light['use_stream_k'] is False
        assert light['block_shape'][2] % light['warp_shape'][2] == 0
    return plan


_base_k_preserving_chooser = choose_partition_configs

def choose_partition_configs(layer, actual_routes, global_experts):
    plan = _base_k_preserving_chooser(layer, actual_routes, global_experts)
    if plan is None or actual_routes <= 4096:
        return plan
    # Experimental padding criterion only; not a native Hopper performance guarantee.
    if all(2 * plan['light'][stage]['block_shape'][0] < plan['heavy'][stage]['block_shape'][0]
           for stage in ('w13', 'w2')):
        plan = deepcopy(plan)
        plan['cutoff'] *= 2
    return plan
