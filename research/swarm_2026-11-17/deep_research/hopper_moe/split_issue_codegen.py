"""Default-off isolated source renderer; no monkeypatches, serving hook or compiler calls."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
PINNED={
 'split_kernel.cuh':'876a75784106ad60dad2af9c534fde1df85d4fff02e96d681380f612ba1934e5',
 'split_control_kernel.cuh':'4952c385403366e654eb7979ea4df026601a66efc6706f78e36c0a77b4753b47',
 'split_wgmma.cuh':'b78c1e5a8ecaf9be25773e57ee6049b22a2dbe4030a159ebe38c5a5fa5246fa4'}
def render_split_issue(generated_source,configuration,sm_version,enabled=False,mode='split'):
    if not enabled:return generated_source,{'enabled':False,'reason':'Default off; original source identity retained'}
    if sm_version!=90 or mode not in ('control','split'):raise ValueError('Unqualified architecture or mode')
    expected={'a_dtype':'bfloat16','b_dtype':'float4e2m1','c_dtype':'bfloat16','bs_dtype':'float8e4m3','weight_scale_group_size':16,'weight_scale_type':'group','weight_scale_2_type':'none','gemm_type':'indexed','mma_type':'wgmma'}
    if any(configuration.get(k)!=v for k,v in expected.items()):raise ValueError('Unsupported datatype/layout/arithmetic')
    for k in ('use_warp_spec','use_stream_k','use_f16_accum','has_zero_point','has_bias','use_packed_k_layout','use_fused_e8m0_scale','use_int_weight_scale'):
        if configuration.get(k,False):raise ValueError('Unsupported '+k)
    if configuration.get('input_scale_group_size',0)!=0:raise ValueError('Input scales unsupported')
    if any(configuration.get(k,1)!=1 for k in ('multi_cast_size_a','multi_cast_size_b')):raise ValueError('Cluster release unqualified')
    if configuration['num_stages'] not in (3,4):raise ValueError('Only compiled stage3/4 family eligible')
    bm,bn,bk=configuration['block_shape'];wm,wn,wk=configuration['warp_shape']
    if wk//16<2 or (wk//16)%2 or bk%wk or bm%wm or bn%wn or (bn//wn)%4:raise ValueError('Unsupported warp geometry')
    if configuration.get('shape_k',0)%bk:raise ValueError('K padding unsupported')
    header='split_kernel.cuh' if mode=='split' else 'split_control_kernel.cuh'
    texts={}
    for f in (header,'split_wgmma.cuh'):
        raw=(P/f).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=PINNED[f]:raise ValueError('Changed unqualified research header: '+f)
        texts[f]=raw.decode('utf-8-sig')
    # Inline research headers so compile cache key contains every changed byte.
    injected=texts[header].replace('#include "split_wgmma.cuh"',texts['split_wgmma.cuh'])
    needle='#include <humming/kernel/humming.cuh>'
    if generated_source.count(needle)!=1:raise ValueError('Unexpected generated source')
    result=generated_source.replace(needle,injected)
    manifest={'enabled':True,'mode':mode,'source_sha256':hashlib.sha256(result.encode()).hexdigest(),'headers':{f:PINNED[f] for f in texts},'qualification':'Offline source prototype only; H200 correctness/performance gates outstanding','extra_persistent_weight_bytes':0}
    return result,manifest
