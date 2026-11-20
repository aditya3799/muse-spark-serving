"""Guarded isolated Humming swap-AB source adapter. No package mutations."""
from pathlib import Path
import hashlib
P=Path(__file__).resolve().parent;INC=P.parents[3]/'swarm_2026-11-16/moe/humming_vendor/humming/include/humming'

def one(s,a,b):
 if s.count(a)!=1:raise ValueError(('source anchor',a,s.count(a)))
 return s.replace(a,b)
def admit(c):
 exact={'a_dtype':'bfloat16','b_dtype':'float4e2m1','c_dtype':'bfloat16','bs_dtype':'float8e4m3','as_dtype':None,'weight_scale_group_size':16,'weight_scale_group_size_n':0,'input_scale_group_size':0,'weight_scale_type':'group','weight_scale_2_type':'none','gemm_type':'indexed','mma_type':'mma','pad_shape_n':0,'pad_shape_k':0,'num_write_splits':1,'num_stages':3,'num_ctas_per_sm':1,'use_cp_async':True}
 if any(c.get(k)!=v for k,v in exact.items()):raise ValueError('Unqualified exact-format/scheduler configuration')
 if any(c.get(k,False)for k in ('use_tma','use_tma_a','use_tma_b','use_tma_bs','use_tma_c','use_tma_as','use_tma_bzp','use_tma_bias','use_tma_bs2','use_warp_spec','use_mbarrier','use_stream_k','use_f16_accum','has_zero_point','is_fp_zero_point','has_bias','use_int_weight_scale','use_fused_e8m0_scale','use_packed_k_layout','use_batch_invariant')):raise ValueError('Unsupported extension')
 if any(c.get(k,1)!=1 for k in ('multi_cast_size_a','multi_cast_size_b','raster_group_m')):raise ValueError('No cluster or grouped raster admission')
 bm,bn,bk=c['block_shape'];wm,wn,wk=c['warp_shape']
 if bm!=wm or bm not in (8,16,24) or bn!=256 or wn!=64 or (bk,wk)not in((64,32),(64,64),(128,64)):raise ValueError('Unqualified swap geometry')
 if c['shape_n']not in(256,4096,8192)or c['shape_k']not in(128,4096)or c['num_experts']not in(1,8):raise ValueError('Fixed research shape only')
 return True

def render(source,c,semistatic=True,direct=False):
 admit(c)
 if type(semistatic)!=bool or type(direct)!=bool:raise ValueError('Explicit schedule and decode bool required')
 swap=(P/'swap_ab_humming_v3.cuh').read_text(encoding='utf-8-sig').replace('@SEMISTATIC@',str(semistatic).lower()).replace('@DIRECT@',str(direct).lower())
 writer=(INC/'epilogue/smem_writer.cuh').read_text()
 writer=one(writer,'static constexpr bool kUseWgmma = Ctx::kUseWgmma;','static constexpr bool kUseWgmma = true; // Local C-fragment orientation only; Ctx stays MMA.')
 start=writer.index('        constexpr uint32_t inner_m = ')
 end=writer.index('\n      }\n    }\n  }\n};',start)
 writer=writer[:start]+'        PRAGMA_UNROLL\n        for(uint32_t p=0;p<2;p++)write_to_smem(part_regs[p],2*i+p,j);'+writer[end:]
 ep=(INC/'epilogue/pipeline.cuh').read_text();ep=one(ep,'#include <humming/epilogue/smem_writer.cuh>',writer)
 s2r=(INC/'memory/s2r_pipeline.cuh').read_text()
 s2r=one(s2r,'loader_a.load(smem.stages[stage_id].a, mma.regs_a_as_ptr(buffer_id), iter_id, stage_id);','mma.load_activation(smem.stages[stage_id].a, buffer_id, iter_id, stage_id);')
 kernel=(INC/'kernel/humming.cuh').read_text()
 kernel=one(kernel,'#include <humming/epilogue/pipeline.cuh>',ep)
 kernel=one(kernel,'#include <humming/memory/s2r_pipeline.cuh>',s2r)
 kernel=one(kernel,'#include <humming/mma/all.cuh>','#include <humming/mma/all.cuh>\n'+swap)
 kernel=one(kernel,'using MMA = Mma<Ctx, MainloopArithmetic>;','using MMA = ResearchSwapMMA<Ctx, MainloopArithmetic>;')
 if direct:
  arith=(INC/'arith/mainloop_arith.cuh').read_text()
  arith=one(arith,'scalar_t2 bs_single = bs_f16_ptr[kUseWgmma ? k : i];','scalar_t2 bs_single = bs_f16_ptr[k]; // Local swap-A scale ownership; all arithmetic unchanged.')
  kernel=one(kernel,'#include <humming/arith/mainloop_arith.cuh>',arith)
 out=one(source,'#include <humming/kernel/humming.cuh>',kernel)
 return out,dict(scope='Indexed cp.async canonical-packed NVFP4 swap-AB, local transposed writer, native arithmetic and K-bank reducer preserved',semistatic=semistatic,direct_canonical=direct,source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in [P/'swap_ab_humming_v3.cuh',P/'swap_ab_humming_codegen_v3.py',INC/'kernel/humming.cuh',INC/'memory/s2r_pipeline.cuh',INC/'epilogue/pipeline.cuh',INC/'epilogue/smem_writer.cuh',INC/'epilogue/smem_reducer.cuh',INC/'arith/mainloop_arith.cuh']})
