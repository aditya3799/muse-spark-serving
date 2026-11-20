# okc: problem=muse_moe_w4a16
# Uses Apache-2.0 licensed Humming and SGLang public APIs.
"""NVFP4 W4A16 expert GEMMs with FP32 accumulation and no Stream-K.

One canonical packed weight representation per expert. Global scales are
applied after each BF16 GEMM output, matching the stock Marlin boundary.
"""
from typing import TypedDict
from types import SimpleNamespace
import torch
try:
    from humming.layer import HummingMethod
except ImportError as exc:
    raise RuntimeError("This candidate requires the judge-pinned humming-kernels 0.1.12 dependency") from exc
try:
    from sglang.kernels.ops.moe.muse_spark_post_expert_finalize_triton import fused_post_expert_norm_weighted_sum
except ImportError:
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum

class Weights(TypedDict):
    w13: torch.Tensor
    w13_sf: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_gs: torch.Tensor
    post_norm_w: torch.Tensor

class Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str

class Inputs(TypedDict):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    config: Config

def prepare_humming(raw,torch):
    from humming import dtypes
    from humming.config import WeightScale2Type,WeightScaleType
    from humming.layer import HummingMethod
    from humming.schema import HummingInputSchema,HummingWeightSchema
    layer=torch.nn.Module()
    for prefix in ('w13','w2'):
        w=getattr(raw,prefix+'_weight');sf=getattr(raw,prefix+'_weight_scale')
        gs=getattr(raw,prefix+'_weight_scale_2')
        e,n,k2=w.shape
        layer.register_parameter(prefix+'_weight',torch.nn.Parameter(w.view(torch.int32),requires_grad=False))
        layer.register_parameter(prefix+'_weight_scale',torch.nn.Parameter(sf,requires_grad=False))
        # Marlin first rounds the accumulator to BF16, then multiplies by a BF16
        # global scale. Native Humming's FP32 epilogue scale has another rounding
        # boundary, so apply this factor separately after each BF16 GEMM output.
        layer.register_buffer(prefix+'_post_scale',gs.to(torch.bfloat16).contiguous())
        schema=HummingWeightSchema(b_dtype=dtypes.float4e2m1,bs_dtype=dtypes.float8e4m3,
                weight_scale_group_size=16,weight_scale_type=WeightScaleType.GROUP,
                weight_scale_2_type=WeightScale2Type.NONE)
        meta=HummingMethod.prepare_layer_meta(layer,shape_n=n,shape_k=k2*2,weight_schema=schema,
                input_schema=HummingInputSchema(a_dtype=dtypes.bfloat16),num_experts=e,
                torch_dtype=torch.bfloat16,sublayer_name=prefix)
        assert meta.a_dtype==dtypes.bfloat16 and meta.c_dtype==dtypes.bfloat16
        assert not meta.use_int_weight_scale and not meta.has_input_scale
        HummingMethod.transform_humming_layer(layer,sublayer_name=prefix)
    layer.register_buffer('locks',torch.zeros(max(2048,torch.cuda.get_device_properties(0).multi_processor_count*8),device='cuda',dtype=torch.int32))
    import copy
    layer.fp32_configs={}
    for stage in ('w13','w2'):
        default=HummingMethod.get_default_tuning_configs(layer,use_f16_accum=False,gemm_type='indexed',sublayer_name=stage)
        snapshot=copy.deepcopy(default)
        owned=copy.deepcopy(default)
        for lo,hi,c in owned:
            c['use_stream_k']=False
            c['use_f16_accum']=False
        assert default==snapshot, 'Default tuning cache must stay unchanged'
        layer.fp32_configs[stage]=owned
    return layer


def indexed(layer,x,local_ids,torch,retain=False):
    from humming.layer import HummingMethod
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.kernels.ops.activation.activation import silu_and_mul
    configs=layer.fp32_configs
    valid=local_ids.numel()
    selected={s:next(c for lo,hi,c in cfg if lo<valid<=hi) for s,cfg in configs.items()}
    assert all(c['use_stream_k'] is False and c['use_f16_accum'] is False for c in selected.values())
    block=selected['w13']['block_shape'][0]
    sorted_ids,expert_ids,padded=moe_align_block_size(local_ids,block,layer.w13_post_scale.numel(),ignore_invalid_expert=True)
    m,k=x.shape;topk=local_ids.shape[1];i=layer.humming_metas['w2'].shape_k
    cache13=torch.empty((m*topk*max(2*i,k),),device=x.device,dtype=x.dtype)
    cache1=cache13[:m*topk*2*i].view(m*topk,2*i)
    common=dict(sorted_ids=sorted_ids,expert_ids=expert_ids,num_tokens_padded=padded,
                valid_shape_m=valid,compute_config={'gemm_type':'indexed','use_f16_accum':False})
    HummingMethod.forward_layer(layer,inputs=x,outputs=cache1,top_k=topk,sublayer_name='w13',tuning_config=selected['w13'],**common)
    safe=local_ids.flatten().clamp_min(0).long()
    cache1.mul_(layer.w13_post_scale[safe,None])
    cache2=torch.empty((m*topk,i),device=x.device,dtype=x.dtype)
    silu_and_mul(cache1,cache2,expert_ids=local_ids.flatten(),expert_step=1)
    cache3=cache13[:m*topk*k].view(m*topk,k)
    if selected['w2']['block_shape'][0]!=block:
        second=moe_align_block_size(local_ids,selected['w2']['block_shape'][0],layer.w2_post_scale.numel(),ignore_invalid_expert=True)
        common.update(sorted_ids=second[0],expert_ids=second[1],num_tokens_padded=second[2])
    HummingMethod.forward_layer(layer,inputs=cache2,outputs=cache3,top_k=1,sublayer_name='w2',tuning_config=selected['w2'],**common)
    cache3.mul_(layer.w2_post_scale[safe,None])
    return cache3.view(m,topk,k),local_ids


def prepare(weights):
    # The task supplies global weights before a per-call shard config. Store each
    # expert once, so any contiguous shard can be selected without caching inputs.
    raw=SimpleNamespace()
    for prefix in ('w13','w2'):
        setattr(raw,prefix+'_weight',weights[prefix])
        setattr(raw,prefix+'_weight_scale',weights[prefix+'_sf'])
        setattr(raw,prefix+'_weight_scale_2',weights[prefix+'_gs'])
    layer=prepare_humming(raw,torch)
    return {'layer':layer,'norm_w':(weights['post_norm_w'].float()+1.0).to(torch.bfloat16)}


def kernel_fn(data,state):
    cfg=data['config']; x=data['hidden_states'].contiguous()
    ids=data['topk_ids'].to(torch.int32).contiguous()
    active=(ids>=cfg['local_start']) & (ids<cfg['local_start']+cfg['local_count'])
    global_ids=torch.where(active,ids,-1)
    outputs,valid_ids=indexed(state['layer'],x,global_ids,torch)
    return fused_post_expert_norm_weighted_sum(outputs,data['topk_weights'].to(torch.float32).contiguous(),
        norm_weight=state['norm_w'],eps=1e-8,weight_first=True,slot_expert_ids=valid_ids)

