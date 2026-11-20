"""Isolated compact fused-W13 renderer; no active source edits."""
from pathlib import Path
import hashlib
from compact_codegen import PATCHED
P=Path(__file__).resolve().parent;INC=P.parents[3]/'swarm_2026-11-16/moe/humming_vendor/humming/include/humming'
ROOT_FUSED=P.parents[1]/'coordination/w13_fused_epilogue.py'
ROOT_FUSED_SHA='1351d9b12d969f96b7ce3c582d2a8e7e27be3e3a70136904454c23dc7375990e'
if hashlib.sha256(ROOT_FUSED.read_bytes()).hexdigest()!=ROOT_FUSED_SHA:raise ValueError('Root fusion source provenance mismatch')
def admit(c):
 if c.get('gemm_type')!='grouped_contiguous' or c.get('use_stream_k') or c.get('use_f16_accum') or c.get('use_tma_c'):raise ValueError('Grouped contiguous, FP32 and legacyC only')
 if c.get('raster_group_m',1)!=1:raise ValueError('Monotonic grouped scheduling required')
 expected={'a_dtype':'bfloat16','c_dtype':'bfloat16','b_dtype':'float4e2m1','bs_dtype':'float8e4m3','weight_scale_group_size':16,'weight_scale_2_type':'none','pad_shape_n':0,'pad_shape_k':0,'shape_n':8192,'shape_k':4096}
 if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Unqualified fused arithmetic/layout')
 if any(c.get(k,False) for k in ('has_bias','has_zero_point','use_int_weight_scale','use_fused_e8m0_scale')):raise ValueError('Unqualified arithmetic extension')
 if c.get('use_warp_spec') and c.get('num_stages',0)<3:raise ValueError('Native WS stage lifetime unqualified')
 return True
def replace_once(text,old,new):
 if text.count(old)!=1:raise ValueError(('source contract',old,text.count(old)))
 return text.replace(old,new)

def render_compact_fused(source,c):
 admit(c)
 writer=(INC/'epilogue/gmem_writer.cuh').read_text()
 writer=writer.replace('int4 *gmem_ptr','uint2 *gmem_ptr').replace('reinterpret_cast<int4 *>','reinterpret_cast<uint2 *>')
 writer=replace_once(writer,'EpilogueGmemWriter(Ctx &ctx, ArithClass &arith) : ctx(ctx), arith(arith) {','EpilogueGmemWriter(Ctx &ctx, ArithClass &arith, const __nv_bfloat16 *scales) : ctx(ctx), arith(arith), post_scales(scales) {\n    static_assert(!kUseStreamK && !kUseTmaC && (kIsIndexedGemm || Ctx::kIsGroupedContiguousGemm) && PadShape::N == 0);')
 writer=replace_once(writer,'  uint32_t row_offset;','  const __nv_bfloat16 *post_scales;\n  float post_scale;\n  CUDA_INLINE void set_expert(uint32_t expert) { post_scale = __bfloat162float(post_scales[expert]); }\n  uint32_t row_offset;')
 start=writer.index('        if (!kUseStreamK || slice_count == 1 || slice_id == 0) {')
 end=writer.index('\n      };',start)
 writer=writer[:start]+'''        uint2 output;
        auto *input_bf16 = reinterpret_cast<__nv_bfloat16 *>(&val);
        auto *output_bf16 = reinterpret_cast<__nv_bfloat16 *>(&output);
        PRAGMA_UNROLL
        for (uint32_t pair = 0; pair < 4; ++pair) {
          float gate_product, up_product;
          float gate = __bfloat162float(input_bf16[2 * pair]);
          float up = __bfloat162float(input_bf16[2 * pair + 1]);
          asm("mul.rn.f32 %0, %1, %2;" : "=f"(gate_product) : "f"(gate), "f"(post_scale));
          asm("mul.rn.f32 %0, %1, %2;" : "=f"(up_product) : "f"(up), "f"(post_scale));
          gate = __bfloat162float(__float2bfloat16_rn(gate_product));
          up = __bfloat162float(__float2bfloat16_rn(up_product));
          float activation = gate / (1.0f + expf(-gate));
          output_bf16[pair] = __float2bfloat16_rn(activation * up);
        }
        gmem_ptr[gmem_offset] = output;'''+writer[end:]
 pipeline=(INC/'epilogue/pipeline.cuh').read_text()
 pipeline=replace_once(pipeline,'#include <humming/epilogue/gmem_writer.cuh>','#include <cuda_bf16.h>\n'+writer)
 pipeline=replace_once(pipeline,'EpiloguePipeline(Ctx &ctx, ArithClass &arith)','EpiloguePipeline(Ctx &ctx, ArithClass &arith, const __nv_bfloat16 *post_scales)')
 pipeline=replace_once(pipeline,'gmem_writer(ctx, arith)','gmem_writer(ctx, arith, post_scales)')
 pipeline=replace_once(pipeline,'    gmem_writer.seek(m_block_id, n_block_id, current_shape_m, m_offset);','    gmem_writer.set_expert(expert_id);\n    gmem_writer.seek(m_block_id, n_block_id, current_shape_m, m_offset);')
 out=source
 for name in ('humming.cuh','humming_ws.cuh'):
  anchor='#include <humming/kernel/'+name+'>'
  if anchor not in out:continue
  kernel=(INC/'kernel'/name).read_text()
  kernel=replace_once(kernel,'#include <humming/scheduler.cuh>',PATCHED)
  kernel=replace_once(kernel,'#include <humming/epilogue/pipeline.cuh>',pipeline)
  kernel=replace_once(kernel,'    bool use_int64_expert_layout) {','    bool use_int64_expert_layout,\n    const __nv_bfloat16 *post_scales) {')
  kernel=replace_once(kernel,'auto epilogue = Epilogue(ctx, epilogue_arith);','auto epilogue = Epilogue(ctx, epilogue_arith, post_scales);')
  out=replace_once(out,anchor,kernel)
 if out==source:raise ValueError('Kernel include not found')
 return out,{'scope':'Compact grouped legacyC fused W13; exact root epilogue arithmetic, endpoint correction, explicit18thscale argument','root_fused_sha256':ROOT_FUSED_SHA,'output_columns':'Nraw/2','tma_c':False}

