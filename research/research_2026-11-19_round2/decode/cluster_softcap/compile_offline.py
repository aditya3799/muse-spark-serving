"""Bounded offline SM90 native/cluster codegen comparison, no CUDA context."""
import ast,hashlib,importlib.util,json,os,resource,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
here=Path(__file__).resolve().parent;root=here.parents[2]
available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
if available<2.3*1024**3:
 (here/'compile_preflight_failure.json').write_text(json.dumps(dict(available=available,required=2.3*1024**3)));print('BLOCKED host memory preflight');sys.exit(2)
resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
import triton
from triton.compiler import ASTSource
from triton.experimental.gluon._runtime import GluonASTSource
from triton.backends.compiler import GPUTarget
source=root/'sglang/python/sglang/kernels/ops/sampling/softcap_row_argmax.py'
text=source.read_text();sha=hashlib.sha256(source.read_bytes()).hexdigest()
proof=json.loads((here.parents[1]/'coordination/target_argmax_postops/correctness.json').read_text())
assert sha==proof['producer_sha256']
fn=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='_softcap_row_argmax_kernel')
(here/'native.py').write_text('import triton\nimport triton.language as tl\nfrom triton.language.extra import libdevice\n@triton.jit\n'+ast.get_source_segment(text,fn)+'\n')
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
native=module('cluster_native',here/'native.py')._softcap_row_argmax_kernel
candidate=module('cluster_candidate',here/'gluon_prototype.py').cluster_softcap
records=[];start=time.monotonic()
for name,func,typ,ctas in [('native',native,ASTSource,1),('cluster',candidate,GluonASTSource,8)]:
 signature={n:t for n,t in zip(func.arg_names[:6],['*fp32','*fp32','i32','i32','i32','fp32'])}
 const={'BLOCK_SIZE':4096} if name=='native' else {}
 kernel=triton.compile(typ(func,signature=signature,constexprs=const,attrs={(0,):[['tt.divisibility',16]],(1,):[['tt.divisibility',16]]}),
  target=GPUTarget('cuda',90,32),options=dict(num_warps=8,num_ctas=ctas))
 for ext in ('ttgir','ptx'):(here/(name+'.'+ext)).write_text(kernel.asm[ext])
 records.append(dict(name=name,metadata=kernel.metadata._asdict(),ptx_sha256=hashlib.sha256(kernel.asm['ptx'].encode()).hexdigest(),ir_sha256=hashlib.sha256(kernel.asm['ttgir'].encode()).hexdigest()))
 print('compiled',name,flush=True)
receipt=dict(complete=True,records=records,available_before=available,elapsed=time.monotonic()-start,native_source_sha256=sha,
 candidate_source_sha256=hashlib.sha256((here/'gluon_prototype.py').read_bytes()).hexdigest(),cuda_devices_hidden=True,
 numerical_proof=False,gpu_execution=False,performance=False,requires='inspect CTA ownership/store predicate/barriers and compare native scalar math; then actual H200 correctness')
(here/'offline_compile.json').write_text(json.dumps(receipt,indent=2,default=str));print('offline complete',receipt['elapsed'])
