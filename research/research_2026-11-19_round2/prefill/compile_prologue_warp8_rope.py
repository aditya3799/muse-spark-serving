"""Single predeclared SM90 BLOCK32/8warp/RoPE layout probe, no CUDA."""
import hashlib,importlib.util,json,os,re,resource,sys,time
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
os.environ['CUDA_VISIBLE_DEVICES']=''
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
H=Path(__file__).resolve().parent;D=H.parent/'decode';O=H/'prologue_warp8';O.mkdir(exist_ok=True)
p=D/'prologue_package/kernel.py';spec=importlib.util.spec_from_file_location('warp8_source',p);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
sig=dict(QKV='*bf16',Q_OUT='*u8',K_OUT='*bf16',POS='*i64',COS_SIN='*fp32',K_CACHE='*u8',V_CACHE='*u8',CACHE_LOC='*i64',eps='fp32',stride_qkv='i32',n_rows='i32',kv_capacity='i32')
constants=dict(NUM_Q_HEADS=16,NUM_KV_HEADS=2,D=64,BD=64,BLOCK_M=32,HAS_ROPE=True,HAS_KV_STORE=True,FP8_KV_STORE=True,USE_PDL=False,FP8_MAX=448.,NATIVE=True)
attrs={(i,):[['tt.divisibility',16]] for i in [0,1,2,3,4,5,6,7,9]}
start=time.monotonic();kernel=triton.compile(ASTSource(mod.fusion18_kernel,signature=sig,constexprs=constants,attrs=attrs),target=GPUTarget('cuda',90,32),options=dict(num_warps=8,debug=True))
ir=kernel.asm['ttgir'];ptx=kernel.asm['ptx'];(O/'sm90_rope_nativefp8.ttgir').write_text(ir);(O/'sm90_rope_nativefp8.ptx').write_text(ptx)
layouts=dict(re.findall(r'^(#\w+) = (#ttg\.blocked<[^\n]+>)$',ir,re.M));mul=next(line for line in ir.splitlines() if '%var = arith.mulf' in line);alias=re.search(r', (#\w+)>',mul).group(1);layout=layouts[alias]
compatible='sizePerThread = [1, 8]' in layout and 'threadsPerWarp = [4, 8]' in layout and 'warpsPerCTA = [8, 1]' in layout
shuffles=[line.strip() for line in ptx.splitlines() if 'shfl.sync.bfly' in line]
out=dict(scope=__doc__,compiled=True,compiler=triton.__version__,elapsed_s=time.monotonic()-start,variance_layout=layout,
 per_head_layout_matches_native=compatible,shuffle_lines=shuffles,metadata=kernel.metadata._asdict(),
 source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),ir_sha256=hashlib.sha256(ir.encode()).hexdigest(),ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
 limitations=['Native4warp versus fused8warp changes row distribution; check full per-head PTX tree','One RoPE specialization only; no H200 execution','No integration or performance claim'])
(O/'offline_rope_receipt.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps(out,indent=2,default=str))
