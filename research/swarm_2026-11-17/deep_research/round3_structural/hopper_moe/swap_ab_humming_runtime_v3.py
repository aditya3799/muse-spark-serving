"""Research SM86 cp.async swap-AB kernel; constructor compiles and needs root lease."""
from humming.kernel.humming import HummingKernel
from humming.config.mma import MmaOpClass
from humming import dtypes
from humming.jit.compiler import NVRTCCompiler
from swap_ab_humming_codegen_v3 import admit,render
class Compiler(NVRTCCompiler):
 @classmethod
 def get_flags(cls,*args,**kwargs):return super().get_flags(*args,**kwargs)+['--ptxas-options=-v']
class SwapHummingKernel(HummingKernel):
 def __new__(cls,*,research_direct=True,research_semistatic=True,**kwargs):
  obj=object.__new__(cls);obj.inited=False;obj.cubin_loaded=False;obj.research_config=dict(kwargs);obj.research_direct=research_direct;obj.research_semistatic=research_semistatic;return obj
 def __init__(self,*,research_direct=True,research_semistatic=True,**kwargs):super().__init__(**kwargs)
 @staticmethod
 def _get_compiler():return Compiler
 def select_mma_op_class(self):
  admit(self.research_config);self.mma_a_dtype=self.a_dtype;self.mma_b_dtype=self.a_dtype
  return MmaOpClass.from_config(self.mma_type,16,8,16,self.a_dtype,self.a_dtype,dtypes.float32)
 def prepare(self):
  assert self.sm_version==86 and not self.use_tma and not self.use_warp_spec and not self.use_mbarrier
  self.code,self.research=render(self.code,self.research_config,self.research_semistatic,self.research_direct);super().prepare()
