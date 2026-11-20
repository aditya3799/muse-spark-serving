"""Direct packed materializer research wrapper. Instantiation compiles/loads; lease required."""
from pathlib import Path
import ctypes,hashlib
P=Path(__file__).resolve().parent

def render_materializer(n,k,source_transpose,target_transpose,fixed_grid_loop):
 if type(n)!=int or type(k)!=int or n<=0 or k<=0 or n%64 or k%64 or n*k>=2**32:raise ValueError('Positive64-aligned uint32 capacity required')
 if any(type(v)!=bool for v in (source_transpose,target_transpose,fixed_grid_loop)):raise ValueError('Explicit bool layout/loop flags required')
 text=(P/'native_materialize.cuh').read_text(encoding='utf-8-sig')
 for token,value in [('@N@',str(n)+'u'),('@K@',str(k)+'u'),('@SRC@',str(source_transpose).lower()),('@DST@',str(target_transpose).lower()),('@LOOP@',str(fixed_grid_loop).lower())]:
  assert text.count(token)==1;text=text.replace(token,value)
 return text

class NativeMaterializer:
 def __init__(self,n,k,*,source_transpose=False,target_transpose=False,fixed_grid_loop=False):
  import torch
  import cuda.bindings.driver as cbd
  from humming.jit.compiler import NVRTCCompiler
  self.n=n;self.k=k;self.fixed_grid_loop=fixed_grid_loop;self.device=torch.cuda.current_device();props=torch.cuda.get_device_properties(self.device)
  if (props.major,props.minor)!=(8,6):raise ValueError('First runtime qualification is local SM86 only')
  self.code=render_materializer(n,k,source_transpose,target_transpose,fixed_grid_loop)
  class Compiler(NVRTCCompiler):
   @classmethod
   def get_flags(cls,*a,**kw):return super().get_flags(*a,**kw)+['--ptxas-options=-v','--fmad=false']
  self.kernel_filename=Compiler.compile(self.code,sm_version='86',kernel_expr='materialize_packed',disable_fast_math=True,postprocess_cubin=None)
  torch.cuda.set_device(self.device)
  result,self.module=cbd.cuModuleLoad(str(self.kernel_filename).encode());assert result==0,repr(result)
  result,self.func=cbd.cuModuleGetFunction(self.module,b'materialize_packed');assert result==0,repr(result)
  self.threads=256;self.blocks=props.multi_processor_count*4 if fixed_grid_loop else (n*k//4+255)//256
  self.source_sha256=hashlib.sha256(self.code.encode()).hexdigest()
 def __call__(self,weight,scale,output):
  import torch
  import cuda.bindings.driver as cbd
  assert weight.dtype==torch.int32 and scale.dtype==torch.float8_e4m3fn and output.dtype==torch.bfloat16
  assert weight.numel()*8==self.n*self.k and scale.numel()*16==self.n*self.k and output.numel()==self.n*self.k
  assert all(t.is_cuda and t.is_contiguous() and t.device==output.device for t in (weight,scale,output))
  assert output.device.index==self.device and torch.cuda.current_device()==self.device
  assert output.data_ptr()%8==0 and weight.data_ptr()%4==0
  out_begin=output.data_ptr();out_end=out_begin+output.numel()*output.element_size()
  for source in (weight,scale):
   source_begin=source.data_ptr();source_end=source_begin+source.numel()*source.element_size()
   assert out_end<=source_begin or source_end<=out_begin, 'Output byte interval overlaps input'
  config=cbd.CUlaunchConfig();config.gridDimX=self.blocks;config.gridDimY=config.gridDimZ=1;config.blockDimX=self.threads;config.blockDimY=config.blockDimZ=1;config.sharedMemBytes=0;config.hStream=torch.cuda.current_stream(self.device).cuda_stream
  values=(weight.data_ptr(),scale.data_ptr(),output.data_ptr());types=(ctypes.c_void_p,)*3
  result,=cbd.cuLaunchKernelEx(config,self.func,(values,types),0);assert result==0,repr(result)
  return output

