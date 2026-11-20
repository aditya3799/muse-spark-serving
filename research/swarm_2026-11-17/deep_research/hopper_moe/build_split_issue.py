"""Prepare isolated split-issue/wait prototypes from hash-pinned canonical headers."""
from pathlib import Path
import json,hashlib,copy
P=Path(__file__).resolve().parent;ROOT=P.parents[2];OLD=ROOT/'swarm_2026-11-16/moe/phase4_hour';V=OLD.parent/'humming_vendor/humming/include/humming'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(V/'mma/wgmma.cuh')=='e3b1e50fa31b96a2f86b4dbec7d83db7d940356d2be653876359754514c355be'
w=(V/'mma/wgmma.cuh').read_text();w=w.replace('make_wgmma_smem_desc','research_make_wgmma_smem_desc').replace('struct WGMMA','struct ResearchWGMMA').replace('  WGMMA(Ctx','  ResearchWGMMA(Ctx').replace('  void run(uint32_t stage_id, uint32_t iter_id)', '  void issue(uint32_t stage_id, uint32_t iter_id)')
old='    wgmma_commit();\n    wgmma_wait<0>();'
new='''    wgmma_commit();
  };

  CUDA_INLINE
  void wait_and_promote(uint32_t stage_id, uint32_t iter_id) {
    constexpr uint32_t kNumIters = kUsePackedKLayout ? 1 : (WarpShape::N / (MmaShape::N / 4) / kPackedKFactor);
    constexpr uint32_t kRunKLoop = kUsePackedKLayout ? kNumKSlabs : kPackedKFactor;
    uint32_t delta_m = kUsePackedKLayout ? iter_id : 0;
    uint32_t delta_j = final_regs_c_index() == 0 ? delta_m : 0;
    wgmma_wait<0>();'''
assert w.count(old)==1;w=w.replace(old,new)
(P/'split_wgmma.cuh').write_text('// Research-only split of original issue and wait; native arithmetic unchanged.\n'+w)
k=(V/'kernel/humming.cuh').read_text()
k=k.replace('#include <humming/mma/all.cuh>','#include <humming/mma/all.cuh>\n#include "split_wgmma.cuh"')
k=k.replace('using MMA = Mma<Ctx, MainloopArithmetic>;','''using MMA = ResearchWGMMA<Ctx, MainloopArithmetic>;
  static_assert(Ctx::kUseWgmma && Ctx::kIsIndexedGemm && !Ctx::kUsePackedKLayout);
  static_assert(ElementA::kBits == 16 && ElementB::kBits == 4 && ElementBS::kBits == 8);
  static_assert(Ctx::kWeightScaleGroupSize == 16 && Ctx::kIsGroupWeightScale);
  static_assert(!Ctx::kUseFusedE8m0Scale && !Ctx::kHasZeroPoint && !Ctx::kHasBias);
  static_assert(!TuningConfig::kUseWarpSpec && !TuningConfig::kUseStreamK);
  static_assert(!ComputeConfig::kUseF16Accum && Ctx::kWarpIters >= 2 && Ctx::kWarpIters % 2 == 0);
  static_assert(TuningConfig::kMultiCastSizeA == 1 && TuningConfig::kMultiCastSizeB == 1);''')
a='          s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1);\n          mma.run(stage_id, warp_iter_id);'
assert k.count(a)==1
control='          s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1);\n          mma.issue(stage_id, warp_iter_id);\n          mma.wait_and_promote(stage_id, warp_iter_id);'
split='          mma.issue(stage_id, warp_iter_id);\n          s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1);\n          mma.wait_and_promote(stage_id, warp_iter_id);'
for name,variant in [('split_control_kernel.cuh',control),('split_kernel.cuh',split)]:
 (P/name).write_text(k.replace(a,variant))
sourcejobs=json.loads((OLD/'producer_compile_jobs.json').read_text())['jobs'];jobs=[]
for m in (8,32,128,256):
 for stage in ('w13','w2'):
  old=next(j for j in sourcejobs if j['stage']==stage and any(lo<m*8<=hi for lo,hi in j['intervals']))
  for mode in ('control','split'):
   job=copy.deepcopy(old);job['configuration']=copy.deepcopy(old['baseline']);job['job_id']='%s_m%d_%s'%(stage,m,mode);job['policy']=mode;job['tokens']=m
   jobs.append(job)
(P/'split_issue_jobs.json').write_text(json.dumps({'scope':'Original canonical configs for topk8 token M8/32/128/256, unchanged native MMA and wait0. Control differs only in source API split; candidate moves prefetch between issue and wait.','jobs':jobs},indent=2)+'\n')
s=(P/'offline_occupancy_sm90.py').read_text().replace("P/'occupancy_jobs.json'","P/'split_issue_jobs.json'").replace('default=12','default=16')
s=s.replace("row['shared_bytes_estimated']=k.estimated_smem_size", """assert configuration['a_dtype']=='bfloat16' and configuration['b_dtype']=='float4e2m1' and configuration['bs_dtype']=='float8e4m3'
  header=P/('split_control_kernel.cuh' if policy=='control' else 'split_kernel.cuh')
  assert k.code.count('#include <humming/kernel/humming.cuh>')==1
  generated=k.code.replace('#include <humming/kernel/humming.cuh>', '#include "'+str(header)+'"')
  row['research_header_sha256']=sha(header)
  row['shared_bytes_estimated']=k.estimated_smem_size""")
s=s.replace('(dest/\'kernel.cu\').write_text(k.code)','(dest/\'kernel.cu\').write_text(generated)').replace('hashlib.sha256(k.code.encode())','hashlib.sha256(generated.encode())').replace('Compiler.compile(k.code,','Compiler.compile(generated,')
s=s.replace("files=[OLD/'producer_tuning.py'", "files=[P/'split_wgmma.cuh',P/'split_control_kernel.cuh',P/'split_kernel.cuh',OLD/'producer_tuning.py'")
(P/'offline_split_issue_sm90.py').write_text(s)
print('Prepared',len(jobs),'matched control/split builds. No compilation.')
