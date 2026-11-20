from pathlib import Path
import hashlib
P=Path(__file__).resolve().parent;V=P.parents[3]/'swarm_2026-11-16/moe/humming_vendor/humming/include/humming'
SCHED=(V/'scheduler.cuh').read_text(encoding='utf-8-sig');OLD='ctx.params.shape_m - ctx.smem.expert_offset[kNumExperts - 1]';NEW='ctx.smem.expert_offset[kNumExperts] - ctx.smem.expert_offset[kNumExperts - 1]'
assert SCHED.count(OLD)==1
PATCHED=SCHED.replace(OLD,NEW)
def render_compact(source,c,enabled=False):
 if not enabled:return source,{'enabled':False}
 if c.get('raster_group_m',1)!=1:raise ValueError('Grouped expert cursor requires monotonic M task sequence')
 if c.get('gemm_type')!='grouped_contiguous' or c.get('use_stream_k') or c.get('use_f16_accum'):raise ValueError('Grouped contiguous, no streamK, FP32 accumulation required')
 if c.get('a_dtype')!='bfloat16' or c.get('b_dtype')!='float4e2m1' or c.get('bs_dtype')!='float8e4m3' or c.get('weight_scale_group_size')!=16:raise ValueError('Unqualified arithmetic')
 out=source
 for name in ('humming.cuh','humming_ws.cuh'):
  needle='#include <humming/kernel/'+name+'>'
  if needle not in out:continue
  kernel=(V/'kernel'/name).read_text(encoding='utf-8-sig');assert kernel.count('#include <humming/scheduler.cuh>')==1;kernel=kernel.replace('#include <humming/scheduler.cuh>',PATCHED);out=out.replace(needle,kernel)
 assert out!=source
 return out,{'enabled':True,'scheduler_original_sha256':hashlib.sha256(SCHED.encode()).hexdigest(),'scheduler_fixed_sha256':hashlib.sha256(PATCHED.encode()).hexdigest(),'change':'Last expert count uses GPU offset[E], keeping fixed allocation capacity separate from active row count','arithmetic':'Kernel mainloop, dequant, native MMA, reduction, epilogue and canonical weights unchanged'}
