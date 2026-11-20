"""Two paired SM90 CTA1/CTA2 compile-only jobs; no CUDA context or module load."""
from pathlib import Path
import argparse,hashlib,json,os,re,sys,time
P=Path(__file__).resolve().parent;V=P.parents[3]/'swarm_2026-11-16/moe/humming_vendor'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(V))
pa=argparse.ArgumentParser();pa.add_argument('--out',type=Path,required=True);a=pa.parse_args();a.out.mkdir(parents=True,exist_ok=True);assert not(a.out/'results.json').exists()
import torch
assert not torch.cuda.is_initialized()
from humming.config import LayerConfig,ComputeConfig,TuningConfig
from humming.kernel.humming import HummingKernel
from humming.config.mma import MmaOpClass
from humming import dtypes
from humming.jit.compiler import NVRTCCompiler
from swap_ab_humming_codegen_v4 import render
class Offline(HummingKernel):
 def __new__(cls,*args,**kwargs):return object.__new__(cls)
 def __post_init__(self):
  LayerConfig.__post_init__(self);ComputeConfig.__post_init__(self);TuningConfig.__post_init__(self);self.sm_version=90;self.sm_version_str='90a';self.init_kernel()
 def prepare(self):assert not torch.cuda.is_initialized()
 def load_cubin(self):raise AssertionError('No moduleload')
 @staticmethod
 def current_context():raise AssertionError('No context')
 @staticmethod
 def _ensure_cuda_context():raise AssertionError('No context')
class Swap(Offline):
 def select_mma_op_class(self):
  self.mma_a_dtype=self.a_dtype;self.mma_b_dtype=self.a_dtype
  return MmaOpClass.from_config(self.mma_type,16,8,16,self.a_dtype,self.a_dtype,dtypes.float32)
class Compiler(NVRTCCompiler):
 @classmethod
 def get_flags(cls,*args,**kwargs):return super().get_flags(*args,**kwargs)+['--ptxas-options=-v']
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
files=[Path(__file__),P/'SM90_SWAP_CTA2_JOBS.json',P/'swap_ab_humming.cuh',P/'swap_ab_humming_codegen_v4.py',*sorted((V/'humming/include').rglob('*.cuh')),*sorted((V/'humming').rglob('*.py'))]
r=dict(status='RUNNING_OFFLINE',scope='NoCUDAcontext/moduleload; sameMMAprimitive stage4/K128/WK64 BM8N256E8, onlylaunchbounds/persistentgridCTA1vs2. Resourcecapacity nottiming.',sources={str(f):sha(f)for f in files},cases=[])
def save():(a.out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
for job in json.loads((P/'SM90_SWAP_CTA2_JOBS.json').read_text())['jobs']:
 c=job['configuration'];ker=(Swap if job['swap']else Offline)(**c);assert not ker.use_tma and not ker.use_warp_spec and not ker.use_mbarrier
 assert ker.estimated_smem_size<=227*1024,'Native actual resolvedSMEM budget'
 assert ker.num_threads==256 and ker.estimated_smem_size==82944
 assert c['num_ctas_per_sm']*(ker.estimated_smem_size+1024)<=227*1024
 generated,meta=render(ker.code,c,job['semistatic'],True,sm_version=90)if job['swap']else(ker.code,None)
 dest=a.out/job['label'];dest.mkdir();(dest/'kernel.cu').write_text(generated);(dest/'kernel_expression.txt').write_text(ker.kernel_expr);mma=ker.mma_op_class.to_cpp_str();(dest/'mma.cuh').write_text(mma)
 row=dict(label=job['label'],configuration=c,source_sha256=hashlib.sha256(generated.encode()).hexdigest(),mma_sha256=hashlib.sha256(mma.encode()).hexdigest(),math_threads=ker.num_math_threads,total_threads=ker.num_threads,shared_bytes=ker.estimated_smem_size,metadata=meta,status='compiling');r['cases'].append(row);save();start=time.monotonic()
 cb=Path(Compiler.compile(generated,sm_version='90a',kernel_expr=ker.kernel_expr,disable_fast_math=ker.disable_fast_math,postprocess_cubin=None));(dest/'kernel.sm90a.cubin').write_bytes(cb.read_bytes());logs=''
 for name in ['stdout.log','stderr.log','cmdline.json']:
  if(cb.parent/name).exists():(dest/name).write_bytes((cb.parent/name).read_bytes());logs+=(cb.parent/name).read_text(errors='replace')+'\n'
 row.update(status='compiled',seconds=time.monotonic()-start,cubin_sha256=sha(cb),resources=[x for x in logs.splitlines()if any(t in x for t in ['Used ','spill','stack frame'])],zero_spill_and_stack=not any(int(v)for v in re.findall(r'(\d+) bytes (?:spill stores|spill loads|stack frame)',logs)));regs=int(re.search(r'Used (\d+) registers',logs).group(1));allocated_per_warp=((regs*32+255)//256)*256;row.update(registers=regs,allocated_registers_per_cta=allocated_per_warp*8,full_carveout_CTA2_admissible=allocated_per_warp*8*2<=65536 and 2*(ker.estimated_smem_size+1024)<=227*1024 and row['zero_spill_and_stack']);save();assert not torch.cuda.is_initialized()
assert len(r['cases'])==2 and r['cases'][0]['mma_sha256']==r['cases'][1]['mma_sha256']
r.update(status='COMPLETE_OFFLINE',cuda_initialized=torch.cuda.is_initialized(),sources_unchanged=all(sha(f)==h for f,h in r['sources'].items()));save();assert not r['cuda_initialized']and r['sources_unchanged'];print(json.dumps(dict(status=r['status'],cases=len(r['cases']))),flush=True)
