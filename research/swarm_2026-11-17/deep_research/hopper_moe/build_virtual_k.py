"""Build isolated virtual-K source prototype; never changes serving or installed headers."""
from pathlib import Path
import re,hashlib,json,copy
P=Path(__file__).resolve().parent;OLD=P.parents[2]/'swarm_2026-11-16/moe/phase4_hour';V=OLD.parent/'humming_vendor/humming/include/humming'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(V/'mma/wgmma.cuh')=='e3b1e50fa31b96a2f86b4dbec7d83db7d940356d2be653876359754514c355be'
w=(V/'mma/wgmma.cuh').read_text()
w=w.replace('make_wgmma_smem_desc','virtual_k_make_wgmma_smem_desc').replace('struct WGMMA','struct ResearchVirtualKWGMMA').replace('  WGMMA(Ctx','  ResearchVirtualKWGMMA(Ctx')
w=w.replace('  CRegistersArrayType regs_c[2];','  static constexpr uint32_t kVirtualParts = Ctx::kOriginalKPartitions;\n  CRegistersArrayType regs_c[kVirtualParts];')
w=w.replace('    uint32_t buffer_id = iter_id % 2;','    uint32_t buffer_id = iter_id % 2;\n    constexpr uint32_t kOriginalWarpIters = WarpShape::K / kPartMmaShapeK;\n    const uint32_t virtual_part = iter_id / kOriginalWarpIters;',1)
old='uint32_t k_slab = kUsePackedKLayout ? k : ((iter_id % kItersPerHalf) * kPackedKFactor + k);'
assert w.count(old)==1
w=w.replace(old,'uint32_t k_slab = (iter_id % kOriginalWarpIters) * kPackedKFactor + k;')
w=w.replace('      smem_addr += k_slab * 2 * sizeof(int4) + smem_offset;', '      smem_addr += virtual_part * BlockShape::M * kSwizzleBytes;\n      smem_addr += k_slab * 2 * sizeof(int4) + smem_offset;')
assert w.count('regs_c[0][delta_j + j][0], scale_d')==1
w=w.replace('regs_c[0][delta_j + j][0], scale_d','regs_c[virtual_part][delta_j + j][0], scale_d')
needle='  CUDA_INLINE void may_fence_regs(uint32_t delta_j) {'
reduce='''  CUDA_INLINE void reduce_virtual_partitions() {
    static_assert(final_regs_c_index() == 0 && std::is_same<typename MmaOpClass::ValTypeC, float>::value);
    constexpr uint32_t kScalars = sizeof(CRegistersArrayType) / sizeof(float);
    // Match original shared reducer: 2 -> p0+p1; 4 -> (p0+p2)+(p1+p3).
    PRAGMA_UNROLL
    for (uint32_t step = 1; step < kVirtualParts; step *= 2) {
      PRAGMA_UNROLL
      for (uint32_t part = 0; part < kVirtualParts / (2 * step); part++) {
        float *dst = reinterpret_cast<float *>(&regs_c[part]);
        const float *src = reinterpret_cast<const float *>(&regs_c[part + kVirtualParts / (2 * step)]);
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < kScalars; j++) dst[j] += src[j];
      }
    }
  }

'''
assert needle in w;w=w.replace(needle,reduce+needle)
(P/'virtual_k_wgmma.cuh').write_text(w)
ctx='''#pragma once
#include <humming/utils/all.cuh>
template<class Base>
struct ResearchVirtualKContext : Base {
  using Base::Base;
  using BlockShape = typename Base::BlockShape;
  using WarpShape = typename Base::WarpShape;
  static constexpr uint32_t kOriginalKPartitions = BlockShape::K / WarpShape::K;
  static constexpr uint32_t K_WARPS = 1;
  static constexpr uint32_t kWarpIters = Base::kWarpIters * kOriginalKPartitions;
  CUDA_INLINE uint32_t k_warp_id() { return 0; }
  CUDA_INLINE uint32_t k_warp_offset() { return 0; }
};
'''
(P/'virtual_k_context.cuh').write_text(ctx)
e=(V/'epilogue/pipeline.cuh').read_text().replace('EpiloguePipeline','ResearchVirtualKEpiloguePipeline')
line='    if constexpr (BlockShape::K > WarpShape::K) smem_reducer.reduce(regs_c_ptr);'
assert e.count(line)==1;e=e.replace(line,'    // Virtual banks have already been reduced in exactly the original FP32 tree.')
(P/'virtual_k_epilogue.cuh').write_text(e)
k=(V/'kernel/humming.cuh').read_text()
k=k.replace('#include <humming/epilogue/pipeline.cuh>','#include "virtual_k_epilogue.cuh"')
k=k.replace('#include <humming/mma/all.cuh>','#include <humming/mma/all.cuh>\n#include "virtual_k_context.cuh"\n#include "virtual_k_wgmma.cuh"')
pat=r'  using Ctx = KernelContext<[\s\S]*?      LayerConfig, ComputeConfig, TuningConfig>;'
match=re.search(pat,k);assert match
k=k.replace(match.group(0),match.group(0).replace('using Ctx =','using BaseCtx =')+'\n  using Ctx = ResearchVirtualKContext<BaseCtx>;')
k=k.replace('using MMA = Mma<Ctx, MainloopArithmetic>;','''using MMA = ResearchVirtualKWGMMA<Ctx, MainloopArithmetic>;
  static_assert(Ctx::kUseWgmma && Ctx::kIsIndexedGemm && !Ctx::kUsePackedKLayout);
  static_assert(ElementA::kBits == 16 && ElementB::kBits == 4 && ElementBS::kBits == 8);
  static_assert(Ctx::kWeightScaleGroupSize == 16 && Ctx::kIsGroupWeightScale);
  static_assert(!Ctx::kUseFusedE8m0Scale && !Ctx::kHasZeroPoint && !Ctx::kHasBias);
  static_assert(!TuningConfig::kUseWarpSpec && !TuningConfig::kUseStreamK && !ComputeConfig::kUseF16Accum);
  static_assert(WarpShape::K == 64 && WarpShape::N == 32 && BlockShape::M == WarpShape::M);
  static_assert(Ctx::kOriginalKPartitions == 2 || Ctx::kOriginalKPartitions == 4);
  static_assert(Ctx::kNumMathThreads == (BlockShape::N / WarpShape::N) * 32);
  static_assert(TuningConfig::kMultiCastSizeA == 1 && TuningConfig::kMultiCastSizeB == 1);''')
k=k.replace('using Epilogue = EpiloguePipeline<Ctx, MMA, EpilogueArithmetic>;','using Epilogue = ResearchVirtualKEpiloguePipeline<Ctx, MMA, EpilogueArithmetic>;')
k=k.replace('    epilogue.call(mma.final_regs_c_as_ptr());','    mma.reduce_virtual_partitions();\n    epilogue.call(mma.final_regs_c_as_ptr());')
(P/'virtual_k_kernel.cuh').write_text(k)
sourcejobs=json.loads((OLD/'producer_compile_jobs.json').read_text())['jobs'];jobs=[]
for m,stage,stages in ((8,'w13',4),(8,'w2',4),(32,'w2',3),(128,'w2',4)):
 old=next(j for j in sourcejobs if j['stage']==stage and any(lo<m*8<=hi for lo,hi in j['intervals']))
 conf=copy.deepcopy(old['baseline']);conf['num_stages']=stages
 for mode in (('control','virtual') if m==32 else ('virtual',)):
  job=copy.deepcopy(old);job.update(configuration=conf,policy=mode,tokens=m,job_id='%s_m%d_s%d_%s'%(stage,m,stages,mode));jobs.append(job)
(P/'virtual_k_jobs.json').write_text(json.dumps({'scope':'Compile-only exact partition-bank prototype. Original native warpK64, per-bank K order, conversion and FP32 tree retained; physical threads collapse Kwarps. No runtime or performance validation.','jobs':jobs},indent=2)+'\n')
print('Prepared virtual-K headers and',len(jobs),'jobs; no compilation')
