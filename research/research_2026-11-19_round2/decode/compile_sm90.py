"""Offline SM90 compilation only; no CUDA context, launch, or latency measurement."""
import hashlib,importlib.util,json,os,resource,sys,time
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
os.environ['CUDA_VISIBLE_DEVICES']=''
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('offline_prologue_kernel',here/'prologue_package/kernel.py')
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
out=here/'sm90_compile';out.mkdir(exist_ok=True)
sig=dict(QKV='*bf16',Q_OUT='*u8',K_OUT='*bf16',POS='*i64',COS_SIN='*fp32',K_CACHE='*u8',V_CACHE='*u8',CACHE_LOC='*i64',eps='fp32',stride_qkv='i32',n_rows='i32',kv_capacity='i32')
results=[]
for block,rope,native in [(4,True,True),(8,True,True),(4,False,True),(4,True,False)]:
 constants=dict(NUM_Q_HEADS=16,NUM_KV_HEADS=2,D=64,BD=64,BLOCK_M=block,HAS_ROPE=rope,HAS_KV_STORE=True,FP8_KV_STORE=True,USE_PDL=False,FP8_MAX=448.,NATIVE=native)
 start=time.monotonic()
 kernel=triton.compile(ASTSource(mod.fusion18_kernel,signature=sig,constexprs=constants),target=GPUTarget('cuda',90,32),options=dict(num_warps=4,debug=True))
 name=f'b{block}_rope{int(rope)}_native{int(native)}'
 ptx=kernel.asm['ptx'];(out/(name+'.ptx')).write_text(ptx)
 ttir=kernel.asm['ttir'];(out/(name+'.ttir')).write_text(ttir)
 record=dict(name=name,compile_seconds=time.monotonic()-start,ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
  metadata=kernel.metadata._asdict(),fp8_instructions=[line.strip() for line in ptx.splitlines() if 'cvt.' in line and 'e4m3' in line],
  bf16_rounding_instructions=[line.strip() for line in ptx.splitlines() if 'cvt.rn.bf16' in line],shared_bytes=kernel.metadata.shared)
 results.append(record);print(name,'compiled',record['compile_seconds'],flush=True)
 (out/'receipt.json').write_text(json.dumps(dict(scope=__doc__,source_sha256=hashlib.sha256((here/'prologue_package/kernel.py').read_bytes()).hexdigest(),results=results,complete=len(results)==4),indent=2,default=str))
