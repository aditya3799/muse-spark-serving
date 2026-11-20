"""H200-only loader for two existing fused cubins; root runtime guard is unchanged."""
from pathlib import Path
import hashlib,json,sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'coordination'))
from w13_fused_epilogue import FusedW13Kernel,render
from w13_h200_contract import approved_job,require_h200,validate_tensors
import torch

def prepare_h200(job_id):
 job=approved_job(job_id);c=job['configuration']
 manifest=json.loads((P/'w13_fused_jobs.json').read_text())
 renderer=P.parent/'coordination/w13_fused_epilogue.py'
 if hashlib.sha256(renderer.read_bytes()).hexdigest()!=manifest['renderer_sha256']:raise ValueError('Driver/renderer source changed since audited compilation')
 receipt=json.loads((P/'offline_w13_fused/results.json').read_text())
 row=next(r for r in receipt['cases'] if r['job_id']==job_id)
 if row['configuration']!=c or row['status']!='compiled':raise ValueError('Offline receipt mismatch')
 cubin=P/'offline_w13_fused'/job_id/'kernel.sm90a.cubin'
 if hashlib.sha256(cubin.read_bytes()).hexdigest()!=row['cubin_sha256']:raise ValueError('Compiled cubin changed')
 class H200PrecompiledFusedW13(FusedW13Kernel):
  def __new__(cls,*args,**kwargs):
   obj=object.__new__(cls);obj.inited=False;obj.cubin_loaded=False;return obj
  def prepare(self):
   prop=torch.cuda.get_device_properties();require_h200(prop.major,prop.minor,prop.name)
   assert self.sm_version==90 and self.is_indexed_gemm and not self.use_tma
   assert not self.use_stream_k and not self.use_f16_accum and not self.use_warp_spec
   assert not self.has_bias and not self.has_input_scale and self.use_cp_async
   assert str(self.a_dtype)==str(self.c_dtype)=='bfloat16'
   assert str(self.b_dtype)=='float4e2m1' and str(self.bs_dtype)=='float8e4m3'
   assert self.weight_scale_group_size==16 and all(x==0 for x in self.pad_shape[1:])
   self.code=render(self.code)
   if hashlib.sha256(self.code.encode()).hexdigest()!=row['generated_sha256']:raise ValueError('Rendered source differs from offline compiled source')
   self.kernel_filename=str(cubin);self.load_cubin()
  def run(self,layer,x,out,ids,aligned):
   validate_tensors(layer,x,out,ids,aligned,c)
   return super().run(layer,x,out,ids,aligned)
 return H200PrecompiledFusedW13(**c)
