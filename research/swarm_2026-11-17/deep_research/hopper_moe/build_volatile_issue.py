from pathlib import Path
import json,copy,re
P=Path(__file__).resolve().parent;V=P.parents[2]/'swarm_2026-11-16/moe/humming_vendor/humming/include/humming'
b=(V/'memory/s2r_loader/loader_b.cuh').read_text().replace('S2RMemoryLoaderB','ResearchVolatileLoaderB')
helper='''
// Narrow experiment: identical aligned64bit shared load, volatile PTX semantics.
CUDA_INLINE uint2 research_volatile_shared_load(const uint2 *p) {
  uint2 result;
  uint32_t address=cast_smem_ptr_to_uint(p);
  asm volatile("ld.volatile.shared.v2.u32 {%0, %1}, [%2];" : "=r"(result.x), "=r"(result.y) : "r"(address) : "memory");
  return result;
}
'''
b=b.replace('template <class Ctx>',helper+'\ntemplate <class Ctx>',1)
b=b.replace('  using LoadType = typename LoadTypeChooser<kNumIntsPerThread * 4>::Type;', '  using LoadType = typename LoadTypeChooser<kNumIntsPerThread * 4>::Type;\n  static_assert(WarpShape::N == 32 && ElementB::kBits == 4 && ElementA::kBits == 16 && sizeof(LoadType) == 8 && !Ctx::kUsePackedKLayout);')
# Array reads only; brackets have nested arithmetic but no nested indexing.
b,n=re.subn(r'= smem_ptr_load\[([^;]+)\];',r'= research_volatile_shared_load(smem_ptr_load + (\1));',b)
assert n==3
(P/'volatile_loader_b.cuh').write_text(b)
s=(V/'memory/s2r_pipeline.cuh').read_text().replace('S2RMemoryPipeline','ResearchVolatileS2RPipeline').replace('#include <humming/memory/s2r_loader/loader_b.cuh>','#include "volatile_loader_b.cuh"').replace('using LoaderB = S2RMemoryLoaderB<Ctx>;','using LoaderB = ResearchVolatileLoaderB<Ctx>;')
(P/'volatile_s2r_pipeline.cuh').write_text(s)
for mode in ('control','split'):
 original='split_control_kernel.cuh' if mode=='control' else 'split_kernel.cuh';k=(P/original).read_text()
 k=k.replace('#include <humming/memory/s2r_pipeline.cuh>','#include "volatile_s2r_pipeline.cuh"').replace('using S2RMemoryPipeline = S2RMemoryPipeline<Ctx, MMA, Epilogue>;','using S2RMemoryPipeline = ResearchVolatileS2RPipeline<Ctx, MMA, Epilogue>;')
 (P/('volatile_'+mode+'_kernel.cuh')).write_text(k)
jobs=[]
for j in json.loads((P/'split_issue_jobs.json').read_text())['jobs']:
 if j['stage']=='w13' and j['tokens']==32:
  j=copy.deepcopy(j);j['job_id']+='__volatile_b';jobs.append(j)
assert len(jobs)==2
(P/'volatile_issue_jobs.json').write_text(json.dumps({'scope':'Narrow ordering experiment, W13M32 existingnonvolatile controls; same aligned64bit Bloadaddresses, noadditionalGPUbarrier, noasyncregister dependency. VolatilePTXdoesnotguaranteeorderingvsWGMMA.','jobs':jobs},indent=2)+'\n')
s=(P/'offline_split_issue_sm90.py').read_text().replace("P/'split_issue_jobs.json'","P/'volatile_issue_jobs.json'").replace('default=16','default=2')
s=s.replace("'split_control_kernel.cuh' if policy=='control' else 'split_kernel.cuh'","'volatile_control_kernel.cuh' if policy=='control' else 'volatile_split_kernel.cuh'")
s=s.replace("files=[P/'split_wgmma.cuh'","files=[P/'volatile_loader_b.cuh',P/'volatile_s2r_pipeline.cuh',P/'volatile_control_kernel.cuh',P/'volatile_split_kernel.cuh',P/'split_wgmma.cuh'")
(P/'offline_volatile_issue_sm90.py').write_text(s)
print('Prepared2volatile ordering controls, notcompiled')
