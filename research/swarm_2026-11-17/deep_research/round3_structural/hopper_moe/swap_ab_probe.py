"""Small exact-format mapping probe; CUDA constructor requires root lease."""
from pathlib import Path
import ctypes,hashlib
P=Path(__file__).resolve().parent

def render(n,k,banks):
 if n!=256 or k not in (128,4096) or banks not in (1,2):raise ValueError('Only fixed reviewed probe shapes')
 text=(P/'swap_ab_probe.cuh').read_text(encoding='utf-8-sig')
 for key,value in [('N',n),('K',k),('BANKS',banks)]:assert text.count('@'+key+'@')==1;text=text.replace('@'+key+'@',str(value))
 return text
class Probe:
 def __init__(self,n,k,banks):
  import torch,cuda.bindings.driver as cbd
  from humming.jit.compiler import NVRTCCompiler
  assert torch.cuda.get_device_capability()==(8,6)
  self.device=torch.cuda.current_device();self.n=n;self.k=k;self.banks=banks;self.code=render(n,k,banks)
  class Compiler(NVRTCCompiler):
   @classmethod
   def get_flags(cls,*a,**kw):return super().get_flags(*a,**kw)+['--ptxas-options=-v','--fmad=false']
  self.kernel_filename=Compiler.compile(self.code,sm_version='86',kernel_expr='swap_ab',disable_fast_math=True,postprocess_cubin=None)
  result,self.module=cbd.cuModuleLoad(str(self.kernel_filename).encode());assert result==0
  self.functions={}
  for name in ('swap_ab','standard_mma','unpack_operand'):
   result,f=cbd.cuModuleGetFunction(self.module,name.encode());assert result==0;self.functions[name]=f
 def launch(self,name,x,w,s,slots,out,raw):
  import torch,cuda.bindings.driver as cbd
  assert name in ('swap_ab','standard_mma')
  assert x.dtype==out.dtype==torch.bfloat16 and raw.dtype==torch.float32 and w.dtype==torch.int32 and s.dtype==torch.float8_e4m3fn and slots.dtype==torch.int32
  assert x.shape==(24,self.k) and slots.shape==(24,) and w.shape==(1,self.k//16,self.n*2) and s.shape==(1,self.k//16,self.n)
  assert out.shape==raw.shape==(24*8,self.n)
  ts=(x,w,s,slots,out,raw);assert all(t.is_cuda and t.is_contiguous() and t.device==x.device for t in ts)
  assert x.device.index==self.device and torch.cuda.current_device()==self.device
  spans=[(t.data_ptr(),t.data_ptr()+t.numel()*t.element_size())for t in ts]
  for i in (4,5):
   for j in range(len(ts)):
    if i!=j:assert spans[i][1]<=spans[j][0] or spans[j][1]<=spans[i][0]
  c=cbd.CUlaunchConfig();c.gridDimX=self.n//(16 if name=='swap_ab' else 8);c.gridDimY=c.gridDimZ=1;c.blockDimX=32;c.blockDimY=c.blockDimZ=1;c.sharedMemBytes=0;c.hStream=torch.cuda.current_stream().cuda_stream
  vals=tuple(t.data_ptr()for t in ts)+(24*8,);types=(ctypes.c_void_p,)*6+(ctypes.c_int32,)
  result,=cbd.cuLaunchKernelEx(c,self.functions[name],(vals,types),0);assert result==0
 def unpack(self,w,s,out):
  import torch,cuda.bindings.driver as cbd
  assert out.shape==(self.n,self.k) and out.dtype==torch.bfloat16
  assert w.shape==(1,self.k//16,self.n*2) and w.dtype==torch.int32 and s.shape==(1,self.k//16,self.n) and s.dtype==torch.float8_e4m3fn
  assert all(t.is_cuda and t.is_contiguous() and t.device==out.device for t in (w,s,out))
  assert out.device.index==self.device and torch.cuda.current_device()==self.device
  lo=out.data_ptr();hi=lo+out.numel()*out.element_size()
  for t in (w,s):assert hi<=t.data_ptr() or t.data_ptr()+t.numel()*t.element_size()<=lo
  c=cbd.CUlaunchConfig();c.gridDimX=(self.n*self.k+255)//256;c.gridDimY=c.gridDimZ=1;c.blockDimX=256;c.blockDimY=c.blockDimZ=1;c.sharedMemBytes=0;c.hStream=torch.cuda.current_stream().cuda_stream
  result,=cbd.cuLaunchKernelEx(c,self.functions['unpack_operand'],((w.data_ptr(),s.data_ptr(),out.data_ptr()),(ctypes.c_void_p,)*3),0);assert result==0
