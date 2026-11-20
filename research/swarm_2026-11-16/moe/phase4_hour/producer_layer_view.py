"""Isolated metadata-only layer view for Hopper producer experiments.
Not installed in serving. Unsupported metadata returns the original scored layer.
"""
import copy,json
from producer_tuning import producer_config

def producer_layer_view(layer,policy,sm_version):
    if sm_version != 90 or policy not in ('tma_b','tma_b_only'):
        return layer,{'enabled':False,'reason':'unsupported_architecture_or_unpromoted_policy'}
    expected={'a_dtype':'bfloat16','b_dtype':'float4e2m1','c_dtype':'bfloat16','bs_dtype':'float8e4m3',
              'mma_type':'wgmma','weight_scale_type':'group','weight_scale_group_size':16,
              'weight_scale_2_type':'none','use_int_weight_scale':False,'use_fused_e8m0_scale':False}
    configs={}
    for stage in ('w13','w2'):
        meta=json.loads(layer.humming_metas[stage].to_str())
        if any(meta.get(k)!=v for k,v in expected.items()):
            return layer,{'enabled':False,'reason':'unsupported_weight_metadata','stage':stage}
        if meta.get('has_zero_point') or meta.get('has_bias') or meta.get('use_packed_k_layout') or meta.get('has_input_scale'):
            return layer,{'enabled':False,'reason':'unsupported_optional_metadata','stage':stage}
        configs[stage]=[]
        for lo,hi,original in layer.fp32_configs[stage]:
            config=producer_config(original,policy,sm_version)
            if config is None:return layer,{'enabled':False,'reason':'unsupported_tuning_interval','stage':stage,'lo':lo,'hi':hi}
            configs[stage].append((lo,hi,config))
    view=copy.copy(layer)
    view.fp32_configs=configs
    # The clone is metadata-only: tensor dictionaries and metadata remain shared.
    for name in ('_parameters','_buffers','humming_metas'):
        if hasattr(layer,name):assert getattr(view,name) is getattr(layer,name)
    assert view.fp32_configs is not layer.fp32_configs
    return view,{'enabled':True,'policy':policy,'extra_persistent_weight_bytes':0,'shared_parameter_dictionaries':True,
                 'scope':'Research-only Hopper TMA B or B+BS; warp specialization excluded after spill evidence'}
