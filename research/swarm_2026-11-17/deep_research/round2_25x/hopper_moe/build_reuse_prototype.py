from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;ROOT=P.parents[3];V=ROOT/'swarm_2026-11-16/moe/humming_vendor/humming';OLD=P.parents[1]/'hopper_moe'
wmma=(V/'include/humming/mma/wmma.cuh').read_text(encoding="utf-8-sig")
wmma=wmma.replace('struct WMMA {','struct ResearchVirtualKWMMA {').replace('  WMMA(Ctx &ctx, ArithClass &arith)','  ResearchVirtualKWMMA(Ctx &ctx, ArithClass &arith)')
wmma=wmma.replace('  CRegistersArrayType regs_c[2];','  static constexpr uint32_t kVirtualParts = Ctx::kOriginalKPartitions;\n  CRegistersArrayType regs_c[kVirtualParts];')
wmma=wmma.replace('    uint32_t buffer_id = iter_id % 2;','    uint32_t buffer_id = iter_id % 2;\n    uint32_t virtual_part = iter_id / (WarpShape::K / kPartMmaShapeK);')
wmma=wmma.replace('regs_c[0][m][j], regs_c[0][m][j]','regs_c[virtual_part][m][j], regs_c[virtual_part][m][j]')
wmma=wmma.replace('arith.may_apply_as_and_bs_on_mma_c(regs_c_as_ptr(), m, j, k, iter_id);','static_assert(ElementA::kBits == 16); // Canonical may_apply_as_and_bs_on_mma_c immediately returns for A16.')
reduce=(OLD/'virtual_k_wgmma.cuh').read_text(encoding="utf-8-sig");reduce=reduce[reduce.index('  CUDA_INLINE void reduce_virtual_partitions()'):reduce.index('  CUDA_INLINE void may_fence_regs')]
reduce=reduce.replace('final_regs_c_index() == 0 && ','')
wmma=wmma.replace('  template <class T = uint32_t>\n  CUDA_INLINE T *regs_a_as_ptr',reduce+'\n  template <class T = uint32_t>\n  CUDA_INLINE T *regs_a_as_ptr')
(P/'virtual_k_wmma.cuh').write_text(wmma,encoding='utf-8')
(P/'virtual_k_context.cuh').write_bytes((OLD/'virtual_k_context.cuh').read_bytes())
writer=(V/'include/humming/epilogue/smem_writer.cuh').read_text(encoding='utf-8-sig').replace('class EpilogueSmemWriter','class ResearchVirtualKSmemWriter').replace('  EpilogueSmemWriter(Ctx &ctx, ArithClass &arith)','  ResearchVirtualKSmemWriter(Ctx &ctx, ArithClass &arith)').replace('K_WARPS = BlockShape::K / WarpShape::K','K_WARPS = Ctx::K_WARPS')
(P/'virtual_k_smem_writer.cuh').write_text(writer,encoding='utf-8')
epi=(OLD/'virtual_k_epilogue.cuh').read_text(encoding='utf-8-sig').replace('#include <humming/epilogue/smem_writer.cuh>','#include "virtual_k_smem_writer.cuh"').replace('using SmemWriter = EpilogueSmemWriter<','using SmemWriter = ResearchVirtualKSmemWriter<')
(P/'virtual_k_epilogue.cuh').write_text(epi,encoding='utf-8')
k=(OLD/'virtual_k_kernel.cuh').read_text(encoding="utf-8-sig").replace('virtual_k_wgmma.cuh','virtual_k_wmma.cuh').replace('ResearchVirtualKWGMMA','ResearchVirtualKWMMA')
k=k.replace('Ctx::kUseWgmma && Ctx::kIsIndexedGemm','!Ctx::kUseWgmma && !Ctx::kUseMxmma && Ctx::kIsIndexedGemm')
k=k.replace('WarpShape::K == 64 && WarpShape::N == 32 && BlockShape::M == WarpShape::M','WarpShape::K == 64 && WarpShape::N == 64 && BlockShape::M % WarpShape::M == 0')
k=k.replace('(BlockShape::N / WarpShape::N) * 32','(BlockShape::M / WarpShape::M) * (BlockShape::N / WarpShape::N) * 32')
(P/'virtual_k_kernel.cuh').write_text(k,encoding='utf-8')
s=(OLD/'virtual_k_codegen.py').read_text(encoding="utf-8-sig").replace('virtual_k_wgmma.cuh','virtual_k_wmma.cuh').replace('Compile-only prototype; no native runtime qualification','Research prototype; runtime gate required')
s=s.replace("'virtual_k_epilogue.cuh')","'virtual_k_epilogue.cuh','virtual_k_smem_writer.cuh')")
s=s.replace("'mma_type':'wgmma'","'mma_type':'mma'").replace('sm_version!=90','sm_version!=86')
s=s.replace('bm!=wm or wn!=32 or wk!=64','bm%wm or bm//wm not in (1,2) or wn!=64 or wk!=64').replace(' or bn//wn%4',' or bn%wn')
s=s.replace('accum=wm*parts','accum=wm*wn*parts//32').replace('physical=(bn//wn)*32','physical=(bm//wm)*(bn//wn)*32')
s=s.replace("c['num_stages'] not in (3,4)","c['num_stages'] not in (3,4)")
(P/'virtual_k_codegen.py').write_text(s,encoding='utf-8')
owned=json.loads((OLD/'grid_body1_retry/owned_configs.json').read_text(encoding="utf-8-sig"));c=dict(owned['meta']['w13']);c.pop('sublayer_name');c.update(next(v for lo,hi,v in owned['indexed']['w13'] if lo<64<=hi));c['gemm_type']='indexed';c['use_cp_async']=True;c.pop('num_sms',None)
jobs=[]
for label,enabled,factor in [('baseline',False,1),('virtual',True,1),('virtual_m2',True,2)]:
 cfg=json.loads(json.dumps(c));cfg['block_shape'][0]*=factor
 jobs.append({'label':label,'enabled':enabled,'configuration':cfg})
(P/'SM86_REUSE_JOBS.json').write_text(json.dumps({'scope':'Exact per-row native MMA and K partition banks; virtual_m2 adds an M warp to share each loaded B/scales stage across two row tiles. SM86-only falsification, not H200 evidence.','jobs':jobs,'original_config_source_sha256':hashlib.sha256((OLD/'grid_body1_retry/owned_configs.json').read_bytes()).hexdigest()},indent=2)+'\n',encoding='utf-8')
(P/'launch_probe.py').write_bytes((OLD/'launch_probe.py').read_bytes());(P/'route_contract.py').write_bytes((OLD/'virtual_k_gate_contract.py').read_bytes())
print('Prepared SM86 virtual-K and exact M-warp reuse sources; no compile/runtime.')
