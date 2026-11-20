"""Default-off exact q/4 bit lookup for one qualified native WGMMA family."""
import hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[3];INC=ROOT/'swarm_2026-11-16/moe/humming_vendor/humming/include/humming'
HELPER=r'''
CUDA_INLINE uint32_t research_fp4_quarter_pair(uint32_t val) {
  uint32_t select = ((val >> 12) & 7u) | ((val >> 20) & 0x700u);
  uint32_t lo, hi;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo) : "r"(0xc0800000u), "r"(0xc0804000u), "r"(select));
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi) : "r"(0x3e3e3e00u), "r"(0x3f3f3f3fu), "r"(select));
  return (lo & 0x00ff00ffu) | ((hi & 0x00ff00ffu) << 8) | (val & 0x80008000u);
}
'''
def render_lookup(source,c,sm_version,enabled=False):
 if not enabled:return source,{'enabled':False}
 expected={'a_dtype':'bfloat16','b_dtype':'float4e2m1','bs_dtype':'float8e4m3','c_dtype':'bfloat16','weight_scale_group_size':16,'weight_scale_type':'group','weight_scale_2_type':'none','mma_type':'wgmma','gemm_type':'indexed','pad_shape_n':0,'pad_shape_k':0}
 if sm_version!=90 or any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Unqualified dtype/geometry')
 if c.get('num_stages') not in (3,4) or c['warp_shape'][2]!=64 or c['shape_k']%c['block_shape'][2]:raise ValueError('Unqualified K/stages')
 for k in ('use_warp_spec','use_stream_k','use_f16_accum','has_zero_point','has_bias','use_packed_k_layout','use_fused_e8m0_scale','use_int_weight_scale'):
  if c.get(k,False):raise ValueError('Unsupported '+k)
 if any(v for k,v in c.items() if k.startswith('use_tma')) or not c.get('use_cp_async',False):raise ValueError('Require nonTMA cp.async')
 if c.get('input_scale_group_size',0) or c.get('as_dtype') or c.get('num_write_splits',1)!=1 or any(c.get(k,1)!=1 for k in ('multi_cast_size_a','multi_cast_size_b')):raise ValueError('Unsupported scales/cluster')
 main=(INC/'arith/mainloop_arith.cuh').read_text();mma=(INC/'mma/wgmma.cuh').read_text();allmma=(INC/'mma/all.cuh').read_text();kernel=(INC/'kernel/humming.cuh').read_text()
 old='if constexpr (ElementA::kBits == 16 && kExpOffset.x) {';assert main.count(old)==1
 new='static_assert(std::is_same<ElementA, BFloat16>::value && std::is_same<ElementB, Float4E2M1>::value && std::is_same<ElementBS, Float8E4M3>::value && kWeightScaleGroupSize == 16 && kExpOffset.x == 124 && kExpOffset.y == 120, "Exact quarter lookup contract");\n    if constexpr (false && ElementA::kBits == 16 && kExpOffset.x) {'
 main=main.replace(old,new)
 needle='dequant<ElementB, ElementA, kHasZeroPoint, kIsFpZeroPoint, kNumWarpShapeNSplits>(regs_qb[buffer_id], regs_b_ptr, i, zp_vals_ptr);';assert mma.count(needle)==1
 repl='PRAGMA_UNROLL\n        for (uint32_t pair = 0; pair < 4; ++pair) {\n          uint32_t val = regs_qb[buffer_id][i] << (pair * 4);\n          regs_b_ptr[3 - pair] = research_fp4_quarter_pair(val);\n        }'
 mma=mma.replace(needle,repl);mma=mma.replace('template <class Ctx, class ArithClass>',HELPER+'\ntemplate <class Ctx, class ArithClass>',1)
 assert 'research_fp4_quarter_pair(uint32_t val)' in mma
 def replace_once(text,old,new):
  assert text.count(old)==1,(old,text.count(old));return text.replace(old,new)
 allmma=replace_once(allmma,'#include <humming/mma/wgmma.cuh>',mma)
 kernel=replace_once(kernel,'#include <humming/arith/mainloop_arith.cuh>',main)
 kernel=replace_once(kernel,'#include <humming/mma/all.cuh>',allmma)
 out=replace_once(source,'#include <humming/kernel/humming.cuh>',kernel)
 return out,{'enabled':True,'quarter_lut':[0,0x3e00,0x3e80,0x3ec0,0x3f00,0x3f40,0x3f80,0x3fc0],'changes':['FP4 generic decode replaced by signed quarter lookup','Only per-weight exponent HMUL2 removed; existing BS conversion/group multiply/offsets/MMA/epilogue unchanged'],'source_hashes':{str(x.relative_to(INC)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [INC/'arith/mainloop_arith.cuh',INC/'mma/wgmma.cuh',INC/'mma/all.cuh',INC/'kernel/humming.cuh']}}
