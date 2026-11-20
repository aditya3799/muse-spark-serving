"""Isolated W13 epilogue prototype with an explicit extra scale argument and N/2 output contract."""
import pathlib,ctypes,json,hashlib
import torch
import cuda.bindings.driver as cbd
from humming.kernel.humming import HummingKernel
from humming.config import WeightScale2Type
from humming.jit.runtime import KernelRuntime
ROOT=pathlib.Path(__file__).resolve().parents[3]
INC=ROOT/'swarm_2026-11-16/moe/humming_vendor/humming/include/humming'

def replace_once(text,old,new):
 if text.count(old)!=1:raise ValueError(('source contract',old,text.count(old)))
 return text.replace(old,new)

def render(source):
 writer=(INC/'epilogue/gmem_writer.cuh').read_text()
 writer=writer.replace('int4 *gmem_ptr','uint2 *gmem_ptr').replace('reinterpret_cast<int4 *>','reinterpret_cast<uint2 *>')
 writer=replace_once(writer,'EpilogueGmemWriter(Ctx &ctx, ArithClass &arith) : ctx(ctx), arith(arith) {','EpilogueGmemWriter(Ctx &ctx, ArithClass &arith, const __nv_bfloat16 *scales) : ctx(ctx), arith(arith), post_scales(scales) {\n    static_assert(!kUseStreamK && !kUseTmaC && kIsIndexedGemm && PadShape::N == 0);')
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
 kernel=(INC/'kernel/humming.cuh').read_text()
 kernel=replace_once(kernel,'#include <humming/epilogue/pipeline.cuh>',pipeline)
 kernel=replace_once(kernel,'    bool use_int64_expert_layout) {','    bool use_int64_expert_layout,\n    const __nv_bfloat16 *post_scales) {')
 kernel=replace_once(kernel,'auto epilogue = Epilogue(ctx, epilogue_arith);','auto epilogue = Epilogue(ctx, epilogue_arith, post_scales);')
 return replace_once(source,'#include <humming/kernel/humming.cuh>',kernel)

class FusedW13Kernel(HummingKernel):
 def prepare(self):
  assert self.sm_version==86, 'Runtime probe currently restricted to local SM86'
  assert self.is_indexed_gemm and not self.use_stream_k and not self.use_f16_accum
  assert not self.use_warp_spec and not self.use_tma and not self.has_bias and not self.has_input_scale and not self.has_zero_point
  assert str(self.a_dtype)==str(self.c_dtype)=='bfloat16'
  assert str(self.b_dtype)=='float4e2m1' and str(self.bs_dtype)=='float8e4m3'
  assert self.weight_scale_group_size==16 and self.weight_scale_2_type==WeightScale2Type.NONE
  assert self.pad_shape[1]==0 and self.pad_shape[2]==0
  assert not self.use_int_weight_scale and not self.use_fused_e8m0_scale
  self.code=render(self.code)
  KernelRuntime.prepare(self)
 def load_cubin(self):
  KernelRuntime.load_cubin(self)
  result,pointer,size=cbd.cuModuleGetGlobal(self.module,b'SMEM_SIZE');assert result==0 and size==4
  value=ctypes.c_uint32();result,=cbd.cuMemcpyDtoH(ctypes.addressof(value),pointer,4);assert result==0
  self.dynamic_shared=value.value
  result,=cbd.cuFuncSetAttribute(self.func,cbd.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,self.dynamic_shared);assert result==0
 def run(self,layer,x,out,ids,aligned):
  m,k=x.shape;topk=ids.shape[1];n=self.problem_shape[1]
  assert x.dtype==out.dtype==torch.bfloat16 and x.is_contiguous() and out.is_contiguous()
  assert out.shape==(m*topk,n//2) and k==self.problem_shape[2]
  assert layer.w13_post_scale.shape==(self.num_experts,) and layer.w13_post_scale.dtype==torch.bfloat16
  assert all(t.device==x.device for t in (out,ids,*aligned,layer.w13_post_scale,layer.w13_weight,layer.w13_weight_scale,layer.locks))
  assert out.data_ptr()%8==0 and x.data_ptr()%16==0
  assert layer.w13_weight.data_ptr()%16==0 and layer.w13_weight_scale.data_ptr()%16==0
  assert ids.dtype==torch.int32 and ids.is_contiguous()
  assert all(t.dtype==torch.int32 and t.is_contiguous() for t in aligned)
  assert layer.w13_post_scale.is_contiguous() and layer.w13_weight.is_contiguous() and layer.w13_weight_scale.is_contiguous()
  config=cbd.CUlaunchConfig();config.gridDimX=torch.cuda.get_device_properties(x.device).multi_processor_count*self.num_ctas_per_sm;config.gridDimY=config.gridDimZ=1
  config.blockDimX=self.num_threads;config.blockDimY=config.blockDimZ=1;config.sharedMemBytes=self.dynamic_shared;config.hStream=torch.cuda.current_stream(x.device).cuda_stream
  values=(x.data_ptr(),layer.w13_weight.data_ptr(),out.data_ptr(),0,layer.w13_weight_scale.data_ptr(),0,0,0,aligned[0].data_ptr(),aligned[1].data_ptr(),aligned[2].data_ptr(),0,0,layer.locks.data_ptr(),m,topk,False,layer.w13_post_scale.data_ptr())
  types=(ctypes.c_void_p,)*14+(ctypes.c_uint32,ctypes.c_uint32,ctypes.c_bool,ctypes.c_void_p)
  result,=cbd.cuLaunchKernelEx(config,self.func,(values,types),0);assert result==0
  return out

def prepare(layer,cfg):
 config=json.loads(layer.humming_metas['w13'].to_str());config.pop('sublayer_name',None);config.update(cfg);config.pop('num_sms',None);config.update(gemm_type='indexed',use_f16_accum=False)
 return FusedW13Kernel(**config)
