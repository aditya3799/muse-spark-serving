"""Source-only SM90 B16 resource preparation, optional leased offline compilation."""
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,sys,time,traceback,inspect,textwrap
P=Path(__file__).resolve().parent;ROOT=P.parents[3];V=ROOT/'swarm_2026-11-16/moe/humming_vendor'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HUMMING_DEBUG_KERNEL']='0';sys.path.insert(0,str(V))
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--compile',action='store_true');ap.add_argument('--reduced-m128',action='store_true');ap.add_argument('--start',type=int,default=0);ap.add_argument('--count',type=int,default=24);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
import torch
assert not torch.cuda.is_initialized()
from humming.config import LayerConfig,ComputeConfig,TuningConfig
from humming.kernel.humming import HummingKernel
from humming.jit.compiler import NVRTCCompiler
class OfflineKernel(HummingKernel):
 def __new__(cls,*args,**kwargs):return object.__new__(cls)
 def __post_init__(self):
  LayerConfig.__post_init__(self);ComputeConfig.__post_init__(self);TuningConfig.__post_init__(self)
  self.sm_version=90;self.sm_version_str='90a';self.init_kernel()
 def prepare(self):assert not torch.cuda.is_initialized()
 def load_cubin(self):raise AssertionError('No cubin load')
 @staticmethod
 def current_context():raise AssertionError('No context')
 @staticmethod
 def _ensure_cuda_context():raise AssertionError('No context')
# Explicit isolated capability extension; every other pinned validation remains intact.
from humming import dtypes
checked=textwrap.dedent(inspect.getsource(HummingKernel.check_dtype))
width_cap='assert self.b_dtype.num_bits <= 8'
assert checked.count(width_cap)==1
checked=checked.replace(width_cap,"assert self.b_dtype.num_bits <= 8 or (self.a_dtype == self.b_dtype == self.c_dtype == self.bs_dtype == dtypes.bfloat16 and self.is_channel_weight_scale and not self.has_zero_point and not self.has_bias and not self.use_f16_accum)")
check_ns={'dtypes':dtypes};exec(compile(checked,'isolated_bf16_check_dtype','exec'),check_ns)
OfflineKernel.check_dtype=check_ns['check_dtype']
class Compiler(NVRTCCompiler):
 @classmethod
 def get_flags(cls,*args,**kwargs):return super().get_flags(*args,**kwargs)+['--ptxas-options=-v']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=P.parent/'hopper_moe/SM90_COMPACT_JOBS.json';source=json.loads(manifest.read_text());jobs=[]
# Large-prefill first; every candidate retains exact block/warp shapes and native MMA code.
for ref in sorted(source['jobs'],key=lambda j:-j['tokens']):
 for stages in (4,3,2):
  cfg=dict(ref['baseline']);cfg.update(b_dtype='bfloat16',bs_dtype='bfloat16',weight_scale_type='channel',weight_scale_group_size=0,num_stages=stages)
  jobs.append({'id':ref['job_id'].replace('_grouped_tma_ws','')+'_b16_s'+str(stages),'configuration':cfg,'baseline':ref['baseline']})
if a.reduced_m128:
 jobs=[j for j in jobs if j['id'].startswith(('w13_m768','w2_m768')) and j['configuration']['num_stages']>=3]
 for job in jobs:
  # Deliberate reduced-M candidate; retain exact N/K/WarpK, compare same-M W4 control.
  for key in ('configuration','baseline'):
   job[key]=json.loads(json.dumps(job[key]));job[key]['block_shape'][0]=128;job[key]['warp_shape'][0]=128
  job['id']=job['id'].replace('m768','wm128')
files=[Path(__file__),manifest,*sorted((V/'humming').rglob('*.py')),*sorted((V/'humming/include').rglob('*.cuh'))]
r={'status':'running','scope':'CPU rendering and optional offline SM90 cubin resource study, no CUDA initialization. Quarter-valued BF16 operands and channel4 preserve pinned exponent boundary; runtime correctness remains untested.','source_hashes':{str(f):sha(f) for f in files},'cases':[],'capability_extension':'Only b_dtype<=8 guard extended to exact A=B=C=BS=BF16, channel scale, no zero point/bias/lower precision. Every other pinned dtype/config validation retained.','check_dtype_source':checked}
def save():(a.out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
for job in jobs[a.start:a.start+a.count]:
 row={'job_id':job['id'],'configuration':job['configuration'],'baseline':job['baseline'],'status':'rendering'};r['cases'].append(row);save()
 try:
  k=OfflineKernel(**job['configuration']);b=OfflineKernel(**job['baseline']);mma=k.mma_op_class.to_cpp_str();bmma=b.mma_op_class.to_cpp_str()
  row.update(shared_bytes_estimated=k.estimated_smem_size,baseline_shared_bytes_estimated=b.estimated_smem_size,mma_identical=mma==bmma,mma_sha256=hashlib.sha256(mma.encode()).hexdigest(),baseline_mma_sha256=hashlib.sha256(bmma.encode()).hexdigest(),threads=k.num_threads)
  assert k.block_shape==b.block_shape and k.warp_shape==b.warp_shape
  assert not k.use_stream_k and not k.use_f16_accum
  dest=a.out/job['id'];dest.mkdir(exist_ok=True);(dest/'kernel.cu').write_text(k.code);(dest/'kernel_expression.txt').write_text(k.kernel_expr);(dest/'mma_op_class.cuh').write_text(mma)
  row['generated_sha256']=hashlib.sha256(k.code.encode()).hexdigest()
  if not row['mma_identical']:row['status']='HELD_CHANGED_MMA_PRIMITIVE'
  elif k.estimated_smem_size>227*1024:row['status']='HELD_SHARED_MEMORY_LIMIT'
  elif not a.compile:row['status']='RENDERED_NOT_COMPILED'
  else:
   row['status']='compiling';save();t=time.monotonic();cubin=Path(Compiler.compile(k.code,sm_version='90a',kernel_expr=k.kernel_expr,disable_fast_math=k.disable_fast_math,postprocess_cubin=None));shutil.copyfile(cubin,dest/'kernel.sm90a.cubin');logs=''
   for name in ('stdout.log','stderr.log','cmdline.json'):
    f=cubin.parent/name
    if f.exists():shutil.copyfile(f,dest/name);logs+=f.read_text(errors='replace')+'\n'
   row.update(status='compiled',seconds=time.monotonic()-t,cubin_bytes=cubin.stat().st_size,cubin_sha256=sha(cubin),resource_lines=[line for line in logs.splitlines() if any(s in line for s in ('Used ','spill','stack frame','registers'))],zero_spill_and_stack=not any(int(v) for v in re.findall(r'(\d+) bytes (?:spill stores|spill loads|stack frame)',logs)))
 except Exception as error:row.update(status='FAILED',error=repr(error),traceback=traceback.format_exc())
 assert not torch.cuda.is_initialized();save();print(json.dumps(row),flush=True)
r.update(status='FINISHED_OFFLINE_ONLY',torch_cuda_initialized=torch.cuda.is_initialized(),sources_unchanged=all(sha(Path(f))==h for f,h in r['source_hashes'].items()));save()
