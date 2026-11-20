"""Default-off compile-only virtual K-bank source renderer. No runtime hooks."""
from pathlib import Path
import re,hashlib
P=Path(__file__).resolve().parent
HEADERS=('virtual_k_kernel.cuh','virtual_k_context.cuh','virtual_k_wmma.cuh','virtual_k_epilogue.cuh','virtual_k_smem_writer.cuh')
def render_virtual_k(source,configuration,sm_version,enabled=False):
 if not enabled:return source,{'enabled':False}
 c=configuration
 expected={'a_dtype':'bfloat16','b_dtype':'float4e2m1','bs_dtype':'float8e4m3','c_dtype':'bfloat16','weight_scale_group_size':16,'weight_scale_type':'group','weight_scale_2_type':'none','mma_type':'mma','gemm_type':'indexed'}
 if sm_version!=86 or any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Unqualified architecture or arithmetic metadata')
 for key in ('use_warp_spec','use_stream_k','use_f16_accum','has_zero_point','has_bias','use_packed_k_layout','use_fused_e8m0_scale','use_int_weight_scale'):
  if c.get(key,False):raise ValueError('Unsupported '+key)
 if c.get('input_scale_group_size',0) or c.get('num_write_splits',1)!=1:raise ValueError('Unsupported scale or epilogue split')
 if any(c.get(k,1)!=1 for k in ('multi_cast_size_a','multi_cast_size_b')):raise ValueError('Cluster protocol unqualified')
 bm,bn,bk=c['block_shape'];wm,wn,wk=c['warp_shape'];parts=bk//wk
 if bm%wm or bm//wm not in (1,2) or wn!=64 or wk!=64 or bk%wk or parts not in (2,4) or bn%wn:raise ValueError('Unsupported virtual-partition geometry')
 if c['shape_k']%bk or c['num_stages'] not in (3,4):raise ValueError('Unqualified depth or K tail')
 accum=wm*wn*parts//32
 if accum+24>255:raise ValueError('Mandatory register spill from accumulator and operand lower bound')
 physical=(bm//wm)*(bn//wn)*32;original=physical*parts
 result=source
 for macro,field in [('NUM_THREADS','kNumThreads'),('NUM_MATH_THREADS','kNumMathThreads'),('NUM_LOAD_THREADS','kNumLoadThreads')]:
  patterns=[(r'(#define HUMMING_'+macro+r') '+str(original)+r'\b',r'\g<1> '+str(physical)),(r'(static constexpr auto '+field+r' = )'+str(original)+r'u;',r'\g<1>'+str(physical)+'u;'),(r'(extern "C" __constant__ uint32_t '+macro+r' = )'+str(original)+r';',r'\g<1>'+str(physical)+';')]
  for pattern,replacement in patterns:
   result,count=re.subn(pattern,replacement,result)
   if count!=1:raise ValueError('Unexpected thread metadata declaration: '+macro)
 texts={f:(P/f).read_text(encoding='utf-8-sig') for f in HEADERS}
 injected=texts['virtual_k_kernel.cuh']
 for name in HEADERS[1:]:
  if injected.count('#include "'+name+'"')!=1:raise ValueError('Unexpected research include')
  injected=injected.replace('#include "'+name+'"',texts[name])
 needle='#include <humming/kernel/humming.cuh>'
 if result.count(needle)!=1:raise ValueError('Unexpected canonical kernel include')
 result=result.replace(needle,injected)
 return result,{'enabled':True,'original_threads':original,'physical_threads':physical,'virtual_partitions':parts,'accumulator_register_lower_bound':accum,'accumulator_and_operand_lower_bound':accum+24,'shared_allocation':'Original block/warp layout retained, including unused reduction space','source_sha256':hashlib.sha256(result.encode()).hexdigest(),'header_sha256':{f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in HEADERS},'qualification':'Research prototype; runtime gate required'}
