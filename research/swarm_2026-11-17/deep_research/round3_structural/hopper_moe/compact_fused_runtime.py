"""Explicit18-argument half-width compact fusion launcher. SM86 only; TMA refused."""
import ctypes
import torch
import cuda.bindings.driver as cbd
from humming.kernel.humming import HummingKernel
from humming.jit.runtime import KernelRuntime
from humming.jit.compiler import NVRTCCompiler
from compact_fused_codegen import render_compact_fused,admit
class Compiler(NVRTCCompiler):
 @classmethod
 def get_flags(cls,*args,**kwargs):return super().get_flags(*args,**kwargs)+['--ptxas-options=-v']
class CompactFusedKernel(HummingKernel):
 def __new__(cls,**kwargs):
  obj=object.__new__(cls);obj.inited=False;obj.cubin_loaded=False;obj.research_config=dict(kwargs);return obj
 @staticmethod
 def _get_compiler():return Compiler
 def prepare(self):
  admit(self.research_config)
  assert self.sm_version==86 and self.is_grouped_contiguous_gemm
  assert not self.use_tma and not self.use_warp_spec and not self.use_tma_c
  self.code,self.research=render_compact_fused(self.code,self.research_config)
  KernelRuntime.prepare(self)
 def load_cubin(self):
  KernelRuntime.load_cubin(self)
  result,pointer,size=cbd.cuModuleGetGlobal(self.module,b'SMEM_SIZE');assert result==0 and size==4
  value=ctypes.c_uint32();result,=cbd.cuMemcpyDtoH(ctypes.addressof(value),pointer,4);assert result==0;self.dynamic_shared=value.value
  result,=cbd.cuFuncSetAttribute(self.func,cbd.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,self.dynamic_shared);assert result==0
 def run(self,layer,x,out,offsets,post_scales,locks):
  # No GPU route count is read on the host; shape_m denotes fixed allocation capacity.
  if getattr(layer,'research_weight_layout',None)!='gate_up_interleaved':raise ValueError('Exactly one interleaved W13 representation required')
  assert x.ndim==out.ndim==2 and x.dtype==out.dtype==torch.bfloat16
  assert x.shape[1]==self.problem_shape[2] and out.shape==(x.shape[0],self.problem_shape[1]//2)
  assert offsets.shape==(self.num_experts+1,) and offsets.dtype==torch.int32
  assert post_scales.shape==(self.num_experts,) and post_scales.dtype==torch.bfloat16
  assert locks.dtype==torch.int32 and layer.weight_scale.dtype==torch.float8_e4m3fn
  assert layer.weight.dtype in (torch.int32,torch.int64,torch.uint8)
  tensors=(x,out,offsets,post_scales,locks,layer.weight,layer.weight_scale)
  assert all(t.is_cuda and t.is_contiguous() and t.device==x.device for t in tensors)
  assert x.untyped_storage().data_ptr()!=out.untyped_storage().data_ptr()
  assert x.data_ptr()%16==0 and out.data_ptr()%8==0 and layer.weight.data_ptr()%16==0 and layer.weight_scale.data_ptr()%16==0
  assert torch.cuda.get_device_capability(x.device)==(8,6) and not self.use_tma
  config=cbd.CUlaunchConfig();config.gridDimX=torch.cuda.get_device_properties(x.device).multi_processor_count*self.num_ctas_per_sm;config.gridDimY=config.gridDimZ=1;config.blockDimX=self.num_threads;config.blockDimY=config.blockDimZ=1;config.sharedMemBytes=self.dynamic_shared;config.hStream=torch.cuda.current_stream(x.device).cuda_stream
  values=(x.data_ptr(),layer.weight.data_ptr(),out.data_ptr(),0,layer.weight_scale.data_ptr(),0,0,0,0,0,0,offsets.data_ptr(),0,locks.data_ptr(),x.shape[0],1,False,post_scales.data_ptr())
  types=(ctypes.c_void_p,)*14+(ctypes.c_uint32,ctypes.c_uint32,ctypes.c_bool,ctypes.c_void_p)
  assert len(values)==len(types)==18
  result,=cbd.cuLaunchKernelEx(config,self.func,(values,types),0);assert result==0
  return out

def admit_both(configs):
 if set(configs)!= {'light','heavy'}:raise ValueError('Both W13 policies required before two-buffer admission')
 for c in configs.values():admit(c)
 return True
