"""Default-off offline-only batched WGMMA wait1 source renderer."""
from pathlib import Path
import hashlib
P=Path(__file__).resolve().parent
HEADERS=('wait1_kernel.cuh','wait1_wgmma.cuh')
def render_wait1(source,c,sm_version,enabled=False):
 if not enabled:return source,{'enabled':False}
 expected={'a_dtype':'bfloat16','b_dtype':'float4e2m1','bs_dtype':'float8e4m3','c_dtype':'bfloat16','weight_scale_group_size':16,'weight_scale_type':'group','weight_scale_2_type':'none','mma_type':'wgmma','gemm_type':'indexed','pad_shape_n':0,'pad_shape_k':0}
 if sm_version!=90 or any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Unqualified metadata')
 if c.get('num_stages') not in (3,4) or c['warp_shape'][2]!=64 or c['shape_k']%c['block_shape'][2]:raise ValueError('Unqualified stage/K tail')
 if not c.get('use_cp_async',False):raise ValueError('cp.async required')
 for key in ('use_warp_spec','use_stream_k','use_f16_accum','has_zero_point','has_bias','use_packed_k_layout','use_fused_e8m0_scale','use_int_weight_scale'):
  if c.get(key,False):raise ValueError('Unsupported '+key)
 if any(v for k,v in c.items() if k.startswith('use_tma')):raise ValueError('TMA not qualified')
 if c.get('input_scale_group_size',0) or c.get('num_write_splits',1)!=1:raise ValueError('Unsupported input scale or split')
 if any(c.get(k,1)!=1 for k in ('multi_cast_size_a','multi_cast_size_b')):raise ValueError('Cluster unsupported')
 txt=(P/HEADERS[0]).read_text(encoding='utf-8-sig');needle='#include "wait1_wgmma.cuh"';assert txt.count(needle)==1;txt=txt.replace(needle,(P/HEADERS[1]).read_text(encoding='utf-8-sig'))
 needle='#include <humming/kernel/humming.cuh>';assert source.count(needle)==1;out=source.replace(needle,txt)
 return out,{'enabled':True,'scope':'Offline-only, CPU protocol model; requires native numerical and hazard gate','header_sha256':{f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in HEADERS}}
