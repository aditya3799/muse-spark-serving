"""Research-only metadata view for pipeline/residency controls; no serving hook."""
from copy import copy,deepcopy
import json

def residency_layer_view(layer, stages, ctas, sm_version, producer='cp_async'):
    if sm_version not in (86,90) or stages not in ((2,3) if sm_version==86 else (3,4)) or ctas not in (1,2):
        raise ValueError('Unqualified architecture/pipeline/residency combination')
    if producer not in ('cp_async','tma_b_only','tma_b') or (producer!='cp_async' and sm_version!=90):
        raise ValueError('TMA producer requires Hopper')
    expected={'a_dtype':'bfloat16','b_dtype':'float4e2m1','c_dtype':'bfloat16',
              'bs_dtype':'float8e4m3','weight_scale_group_size':16,
              'weight_scale_type':'group','weight_scale_2_type':'none',
              'use_int_weight_scale':False,'use_fused_e8m0_scale':False}
    view=copy(layer);view.fp32_configs=deepcopy(layer.fp32_configs)
    changed=[]
    for stage,intervals in view.fp32_configs.items():
        meta=json.loads(layer.humming_metas[stage].to_str())
        if any(meta.get(k)!=v for k,v in expected.items()):
            raise ValueError('Unsupported canonical arithmetic metadata')
        if any(meta.get(k,False) for k in ('has_zero_point','has_bias','use_packed_k_layout','has_input_scale')):
            raise ValueError('Unsupported optional metadata')
        for lo,hi,c in intervals:
            if c.get('use_stream_k') is not False or c.get('use_f16_accum') is not False:
                raise ValueError('Require full FP32 per-tile accumulation')
            if sm_version==90 and c['block_shape'][0]!=8:
                continue
            if sm_version==90:
                expected_geometry={'w13':([8,256,128],[8,32,64]),'w2':([8,128,256],[8,32,64])}
                if stage not in expected_geometry or (c['block_shape'],c['warp_shape'])!=expected_geometry[stage]:
                    raise ValueError('Uncompiled Hopper geometry; leave serving fallback untouched')
                if c.get('num_sms',0)<0:
                    raise ValueError('Invalid SM launch count')
            old=deepcopy(c)
            c['num_stages']=stages;c['num_ctas_per_sm']=ctas
            c.update(use_warp_spec=False,use_tma=producer!='cp_async',
                     use_mbarrier=producer!='cp_async',use_cp_async=True,
                     use_tma_a=False,use_tma_as=False,use_tma_c=False,
                     use_tma_b=producer!='cp_async',use_tma_bs=producer=='tma_b',
                     use_tma_bs2=False,use_tma_bzp=False,use_tma_bias=False,
                     multi_cast_size_a=1,multi_cast_size_b=1)
            for key in ('block_shape','warp_shape','use_stream_k','use_f16_accum'):
                assert c[key]==old[key]
            changed.append({'stage':stage,'lo':lo,'hi':hi,'before':old,'after':deepcopy(c)})
    for name in ('_parameters','_buffers','humming_metas'):
        if hasattr(layer,name):assert getattr(view,name) is getattr(layer,name)
    return view,{'changed':changed,'extra_persistent_weight_bytes':0,
                 'qualification':'Research only; compiler/resource evidence does not establish runtime benefit'}
