"""Offline SM90 indexed-producer compilation. No CUDA context or module load."""
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,sys,time
P=Path(__file__).resolve().parent;V=P.parent/'humming_vendor'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HUMMING_DEBUG_KERNEL']='0';sys.path.insert(0,str(V))
pa=argparse.ArgumentParser();pa.add_argument('--out',type=Path,required=True);pa.add_argument('--policies',nargs='+',default=['baseline','tma_b','both']);a=pa.parse_args();a.out.mkdir(parents=True,exist_ok=True)
import torch
assert not torch.cuda.is_initialized()
from humming.config import LayerConfig,ComputeConfig,TuningConfig,MmaType
from humming.kernel.humming import HummingKernel
from humming.jit.compiler import NVRTCCompiler
from producer_tuning import producer_config
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
class OfflineKernel(HummingKernel):
 def __new__(cls,*a,**k):return object.__new__(cls)
 def __post_init__(self):
  LayerConfig.__post_init__(self);ComputeConfig.__post_init__(self);TuningConfig.__post_init__(self)
  self.sm_version=90;self.sm_version_str='90a';self.init_kernel()
 def prepare(self):assert not torch.cuda.is_initialized()
 def load_cubin(self):raise AssertionError('No cubinload')
 @staticmethod
 def current_context():raise AssertionError('No context')
 @staticmethod
 def _ensure_cuda_context():raise AssertionError('No context')
class Compiler(NVRTCCompiler):
 @classmethod
 def get_flags(cls,*args,**kwargs):return super().get_flags(*args,**kwargs)+['--ptxas-options=-v']
files=[P/'producer_tuning.py',Path(__file__),V/'humming/kernel/humming.py',*sorted((V/'humming/include').rglob('*.cuh'))]
r={'status':'started','scope':'Offline SM90compile/resource evidence only; noCUDAcontext/moduleload/H200numerics/timing','cases':[],'source_hashes':{str(f):sha(f) for f in files}}
def save():
 t=a.out/'results.tmp';t.write_text(json.dumps(r,indent=2)+'\n');t.replace(a.out/'results.json')
save();audit=json.loads((P.parent/'sm90_wgmma_metadata_audit.json').read_text());before_mma={}
for stage,routes in (('w13',8),('w2',1024)):
 entry=audit['stages'][stage];original=next(c for lo,hi,c in entry['intervals'] if lo<routes<=hi)
 for policy in a.policies:
  selected=original if policy=='baseline' else producer_config(original,policy)
  assert selected is not None
  configuration=dict(entry['meta'],**selected);configuration.update(gemm_type='indexed',use_f16_accum=False,use_stream_k=False,use_cp_async=True)
  row={'stage':stage,'routes':routes,'policy':policy,'configuration':configuration,'status':'rendering'};r['cases'].append(row);save()
  k=OfflineKernel(**configuration)
  assert k.mma_type==MmaType.WGMMA and k.is_indexed_gemm and not k.use_tma_a and not k.use_tma_c and not k.use_tma_as
  assert not k.use_stream_k and not k.use_f16_accum and not torch.cuda.is_initialized()
  mma=k.mma_op_class.to_cpp_str();digest=hashlib.sha256(mma.encode()).hexdigest()
  if policy=='baseline':before_mma[stage]=digest
  elif stage in before_mma:assert digest==before_mma[stage],'MMA arithmetic changed'
  dest=a.out/(stage+'_'+policy);dest.mkdir(exist_ok=True)
  (dest/'kernel.cu').write_text(k.code);(dest/'kernel_expression.txt').write_text(k.kernel_expr);(dest/'mma_op_class.cuh').write_text(mma)
  row.update(status='compiling',generated_sha256=hashlib.sha256(k.code.encode()).hexdigest(),mma_sha256=digest,flags=Compiler.get_flags('90a',k.disable_fast_math),resolved={'tma_b':k.use_tma_b,'tma_bs':k.use_tma_bs,'warp_spec':k.use_warp_spec,'math_threads':k.num_math_threads,'load_threads':k.num_load_threads,'total_threads':k.num_threads})
  save();start=time.monotonic()
  try:
   cubin=Path(Compiler.compile(k.code,sm_version='90a',kernel_expr=k.kernel_expr,disable_fast_math=k.disable_fast_math,postprocess_cubin=None))
   shutil.copyfile(cubin,dest/'kernel.sm90a.cubin')
   logs=''
   for name in ('stdout.log','stderr.log','cmdline.json'):
    path=cubin.parent/name
    if path.exists():shutil.copyfile(path,dest/name);logs+=path.read_text(errors='replace')+'\n'
   row.update(status='compiled',seconds=time.monotonic()-start,cubin_bytes=cubin.stat().st_size,cubin_sha256=sha(cubin),resource_lines=[line for line in logs.splitlines() if any(s in line for s in ('Used ','spill','stack frame','registers'))],cached_cubin=str(cubin))
  except Exception as error:
   row.update(status='compile_failed',seconds=time.monotonic()-start,error=repr(error));save();raise
  assert not torch.cuda.is_initialized();save();print(json.dumps({key:row[key] for key in ('stage','policy','status','seconds','cubin_bytes')}),flush=True)
r.update(status='compiled',source_unchanged=all(sha(Path(f))==h for f,h in r['source_hashes'].items()),torch_cuda_initialized=torch.cuda.is_initialized());save();assert r['source_unchanged'] and not r['torch_cuda_initialized']
