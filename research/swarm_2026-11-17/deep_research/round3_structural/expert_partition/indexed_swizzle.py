"""Isolated indexed MoE task-order renderer. No arithmetic or serving hooks."""
from pathlib import Path
import hashlib
P=Path(__file__).resolve().parent;ROOT=P.parents[3];V=ROOT/'swarm_2026-11-16/moe/humming_vendor/humming/include/humming'
OLD='''    if constexpr (kRasterGroupM <= 1 || !kIsDenseGemm) {
      m_id = mn_index / N_BLOCKS;
      n_id = mn_index % N_BLOCKS;
    } else {'''
def render_swizzle(source,c,group,audit=False):
 if group not in (1,2,4,8):raise ValueError('Unqualified raster group')
 if group==1:return source,{'group':1,'enabled':False}
 if c.get('gemm_type')!='indexed' or c.get('use_stream_k') or c.get('use_warp_spec') or c.get('use_f16_accum'):raise ValueError('Indexed non-StreamK non-WS FP32 only')
 if c.get('a_dtype')!='bfloat16' or c.get('b_dtype')!='float4e2m1' or c.get('c_dtype')!='bfloat16' or c.get('bs_dtype')!='float8e4m3' or c.get('weight_scale_group_size')!=16:raise ValueError('Unqualified format')
 for key in ('use_tma','use_tma_a','use_tma_b','use_tma_c','use_packed_k_layout','use_fused_e8m0_scale','has_zero_point','has_bias','use_int_weight_scale'):
  if c.get(key,False):raise ValueError('Unsupported '+key)
 if c.get('input_scale_group_size',0) or any(c.get(k,1)!=1 for k in ('multi_cast_size_a','multi_cast_size_b')):raise ValueError('Unsupported inputscale/cluster')
 sched=(V/'scheduler.cuh').read_text(encoding='utf-8-sig');assert sched.count(OLD)==1
 new='''    if constexpr (kIsIndexedGemm) {
      constexpr uint32_t group_size = GROUP;
      const uint32_t blocks_per_group = group_size * N_BLOCKS;
      const uint32_t group_id = mn_index / blocks_per_group;
      const uint32_t first_m = group_id * group_size;
      const uint32_t group_m = MIN(m_blocks - first_m, group_size);
      const uint32_t local = mn_index - group_id * blocks_per_group;
      // Alignment metadata is sorted by expert. Boundary groups keep original order.
      const bool same_expert = ctx.params.expert_ids_ptr[first_m]
          == ctx.params.expert_ids_ptr[first_m + group_m - 1];
      if (group_m > 1 && same_expert) {
        m_id = first_m + local % group_m;
        n_id = local / group_m;
      } else {
        m_id = mn_index / N_BLOCKS;
        n_id = mn_index % N_BLOCKS;
      }
    } else if constexpr (kRasterGroupM <= 1 || !kIsDenseGemm) {
      m_id = mn_index / N_BLOCKS;
      n_id = mn_index % N_BLOCKS;
    } else {'''.replace('GROUP',str(group))
 patched=sched.replace(OLD,new)
 if audit:
  marker='      map_mn_block(dp_mn_next_index, m_block_id, n_block_id);'
  instrumentation=marker+'''
      // Research ownership audit only: StreamK is disabled and locks is a dedicated counter array.
      if (threadIdx.x == 0) {
        atomicAdd(ctx.params.locks + m_block_id * N_BLOCKS + n_block_id, 1);
        if (ctx.params.expert_ids_ptr[m_block_id]
            != ctx.params.expert_ids_ptr[dp_mn_next_index / N_BLOCKS]) {
          atomicAdd(ctx.params.locks + mn_blocks, 1);
        }
      }'''
  assert patched.count(marker)==1;patched=patched.replace(marker,instrumentation)
 kernel=(V/'kernel/humming.cuh').read_text(encoding='utf-8-sig');needle='#include <humming/scheduler.cuh>';assert kernel.count(needle)==1;kernel=kernel.replace(needle,patched)
 needle='#include <humming/kernel/humming.cuh>';assert source.count(needle)==1
 result=source.replace(needle,kernel)
 return result,{'enabled':True,'group':group,'ownership_audit':audit,'scheduler_original_sha256':hashlib.sha256(sched.encode()).hexdigest(),'scheduler_modified_sha256':hashlib.sha256(patched.encode()).hexdigest(),'contract':'Exact same M/N tile ownership, per-tile K order, operands, epilogue and output. Only CTA task visitation order changes. Cross-expert groups remain original order.'}

def map_task(i,experts,n,group):
 first=(i//(group*n))*group;g=min(len(experts)-first,group);local=i-first*n
 if g>1 and experts[first]==experts[first+g-1]:return first+local%g,local//g
 return i//n,i%n
