"""Matched offline-only SM90 M32 baseline/lookup compilation. No CUDA context."""
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,subprocess,sys,time
P=Path(__file__).resolve().parent;ROOT=P.parents[3];V=ROOT/'swarm_2026-11-16/moe/humming_vendor';sys.path.insert(0,str(V));os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HUMMING_DEBUG_KERNEL']='0'
a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);args=a.parse_args();args.out.mkdir(parents=True,exist_ok=True)
import torch
assert not torch.cuda.is_initialized()
from humming.config import LayerConfig,ComputeConfig,TuningConfig,MmaType
from humming.kernel.humming import HummingKernel
from humming.jit.compiler import NVRTCCompiler
from lookup_codegen import render_lookup
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
class Kernel(HummingKernel):
 def __new__(cls,*a,**kw):return object.__new__(cls)
 def __post_init__(self):
  LayerConfig.__post_init__(self);ComputeConfig.__post_init__(self);TuningConfig.__post_init__(self);self.sm_version=90;self.sm_version_str='90a';self.init_kernel()
 def prepare(self):assert not torch.cuda.is_initialized()
 def load_cubin(self):raise AssertionError('No module load')
 @staticmethod
 def current_context():raise AssertionError('No context')
 @staticmethod
 def _ensure_cuda_context():raise AssertionError('No context')
class Compiler(NVRTCCompiler):
 @classmethod
 def get_flags(cls,*a,**kw):return super().get_flags(*a,**kw)+['--ptxas-options=-v']
manifest=P.parent/'hopper_moe/SM90_COMPACT_JOBS.json';c=next(x['baseline'] for x in json.loads(manifest.read_text())['jobs'] if x['job_id']=='w13_m32_grouped_tma_ws')
files=[Path(__file__),P/'lookup_codegen.py',manifest,V/'humming/kernel/humming.py']+list((V/'humming/include').rglob('*.cuh'))
r={'status':'started','scope':'Offline compilation and static disassembly only; no GPU context, no numerical/performance proof','configuration':c,'cases':[],'sources':{str(p):sha(p) for p in files}}
def save():
 t=args.out/'results.tmp';t.write_text(json.dumps(r,indent=2)+'\n');t.replace(args.out/'results.json')
save();k=Kernel(**c);mma=k.mma_op_class.to_cpp_str();digest=hashlib.sha256(mma.encode()).hexdigest()
assert k.mma_type==MmaType.WGMMA and k.is_indexed_gemm and not k.use_warp_spec and not k.use_tma
for enabled in [False,True]:
 name='lookup' if enabled else 'baseline';dest=args.out/name;dest.mkdir(exist_ok=True);code,meta=render_lookup(k.code,c,90,enabled)
 (dest/'kernel.cu').write_text(code);(dest/'mma_op_class.cuh').write_text(mma);(dest/'kernel_expression.txt').write_text(k.kernel_expr)
 row={'variant':name,'status':'compiling','render':meta,'code_sha256':hashlib.sha256(code.encode()).hexdigest(),'mma_sha256':digest,'shared_bytes':k.estimated_smem_size,'threads':k.num_threads,'flags':Compiler.get_flags('90a',k.disable_fast_math)};r['cases'].append(row);save();start=time.monotonic()
 try:
  cubin=Path(Compiler.compile(code,sm_version='90a',kernel_expr=k.kernel_expr,disable_fast_math=k.disable_fast_math,postprocess_cubin=None));shutil.copyfile(cubin,dest/'kernel.sm90a.cubin');logs=''
  for fn in ['stdout.log','stderr.log','cmdline.json']:
   if (cubin.parent/fn).exists():shutil.copyfile(cubin.parent/fn,dest/fn);logs+=(cubin.parent/fn).read_text(errors='replace')+'\n'
  row.update(status='compiled',seconds=time.monotonic()-start,cubin_sha256=sha(cubin),resource_lines=[x for x in logs.splitlines() if any(v in x for v in ['Used ','spill','stack frame'])])
 except Exception as exc:row.update(status='failed',error=repr(exc),seconds=time.monotonic()-start);save();raise
 row['disassembly_status']='Pending CPU Windows nvdisasm after bounded compile'
 assert not torch.cuda.is_initialized();save();print(json.dumps({'variant':name,'status':row['status'],'seconds':row['seconds'],'resources':row['resource_lines']}),flush=True)
r.update(status='compiled',torch_cuda_initialized=torch.cuda.is_initialized(),sources_unchanged=all(sha(Path(p))==v for p,v in r['sources'].items()));save()

