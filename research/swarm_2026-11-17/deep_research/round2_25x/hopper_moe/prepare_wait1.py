from pathlib import Path
import json,hashlib,copy,ast
P=Path(r'D:\OpenKernelChallenge\serving\swarm_2026-11-17\deep_research\round2_25x\hopper_moe');D=P/'wait1';D.mkdir(exist_ok=True);V=P.parents[3]/'swarm_2026-11-16/moe/humming_vendor/humming/include/humming'
w=(V/'mma/wgmma.cuh').read_text();assert w.count('wgmma_wait<0>();')==1
w=w.replace('make_wgmma_smem_desc','research_wait1_make_smem_desc').replace('struct WGMMA','struct ResearchWait1WGMMA').replace('  WGMMA(Ctx','  ResearchWait1WGMMA(Ctx').replace('wgmma_wait<0>();','wgmma_wait<1>();')
# Leave existing A16 compile-time no-op fence/scale helpers; preserve original arithmetic expression.
needle='  CUDA_INLINE void may_fence_regs(uint32_t delta_j) {'
w=w.replace(needle,'''  CUDA_INLINE void drain_async() {
    wgmma_wait<0>();
    // Prevent compiler movement of accumulator consumption above the final wait.
    fence_regs(regs_c[0]);
  }

'''+needle)
(D/'wait1_wgmma.cuh').write_text(w,encoding='utf-8')
k=(V/'kernel/humming.cuh').read_text().replace('#include <humming/mma/all.cuh>','#include <humming/mma/all.cuh>\n#include "wait1_wgmma.cuh"')
k=k.replace('using MMA = Mma<Ctx, MainloopArithmetic>;','''using MMA = ResearchWait1WGMMA<Ctx, MainloopArithmetic>;
  static_assert(Ctx::kUseWgmma && Ctx::kIsIndexedGemm && !Ctx::kUsePackedKLayout);
  static_assert(ElementA::kBits == 16 && ElementB::kBits == 4 && ElementBS::kBits == 8);
  static_assert(Ctx::kWeightScaleGroupSize == 16 && Ctx::kIsGroupWeightScale);
  static_assert(!Ctx::kUseFusedE8m0Scale && !Ctx::kHasZeroPoint && !Ctx::kHasBias);
  static_assert(!TuningConfig::kUseWarpSpec && !TuningConfig::kUseStreamK && !ComputeConfig::kUseF16Accum);
  static_assert(kNumStages == 3 || kNumStages == 4);
  static_assert(WarpShape::K == 64 && Ctx::kWarpIters == 4);
  static_assert(TuningConfig::kMultiCastSizeA == 1 && TuningConfig::kMultiCastSizeB == 1);''')
assert k.count('    consumer.wait_channel();')==1;k=k.replace('    consumer.wait_channel();','    mma.drain_async();\n    consumer.wait_channel();')
(D/'wait1_kernel.cuh').write_text(k,encoding='utf-8')
code='''"""Default-off offline-only batched WGMMA wait1 source renderer."""
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
'''
(D/'wait1_codegen.py').write_text(code,encoding='utf-8');ast.parse(code)
jobs=[];source=json.loads((P/'sm90_corrected/virtual_k_jobs.json').read_text())['jobs']
for tokens,baseid in [(8,'w13_m8_s4_virtual__writer_fixed'),(32,'w13_m32_actual_control')]:
 orig=next(j for j in source if j['job_id']==baseid)
 for stages in (3,4):
  c=copy.deepcopy(orig['configuration']);c['num_stages']=stages
  for mode in ('control','wait1'):
   jobs.append({'stage':'w13','policy':mode,'job_id':'w13_m%d_s%d_%s'%(tokens,stages,mode),'configuration':c,'baseline':copy.deepcopy(c),'intervals':[]})
(D/'jobs.json').write_text(json.dumps({'status':'CPU_REVIEW_PENDING','jobs':jobs},indent=2)+'\n',encoding='utf-8');print('Prepared wait1 source and8 matched jobs; not compiled')
