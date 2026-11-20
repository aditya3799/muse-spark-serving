"""Bounded SM90 compiler-layout audit; no CUDA context, kernel launch or timing.

Aligned pointers/row stride match native launches. n_rows alignment is varied
only for the large out-of-default-budget BLOCK32 control. This is static IR
evidence, not an exhaustive numerical proof or a serving performance result.
"""
import ast,hashlib,importlib.util,json,os,re,resource,sys,time
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
os.environ['CUDA_VISIBLE_DEVICES']=''
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
here=Path(__file__).resolve().parent;root=here.parents[1];out=here/'reduction_layout_sm90';out.mkdir(exist_ok=True)
parent=root/'research_2026-11-18_rl_prefill/decode_qkv/parent_muse_spark_qk_norm_rope_store.py'
text=parent.read_text();tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_qk_rms_norm_fwd_kernel')
native_text='import triton\nimport triton.language as tl\nFP8_E4M3_MAX=448.0\n@triton.jit\n'+ast.get_source_segment(text,node)+'\n'
(out/'native_kernel.py').write_text(native_text)
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
native=module('layout_native',out/'native_kernel.py')._qk_rms_norm_fwd_kernel
fused=module('layout_fused',here/'prologue_package/kernel.py').fusion18_kernel
base_sig=dict(QKV='*bf16',Q_OUT='*bf16',K_OUT='*bf16',POS='*i64',COS_SIN='*fp32',K_CACHE='*bf16',V_CACHE='*bf16',CACHE_LOC='*i64',eps='fp32',stride_qkv='i32',n_rows='i32',kv_capacity='i32')
records=[];started=time.monotonic()
for block in (4,8,16,32):
 for rope in (False,True):
  for kind in ('native','fused_nativefp8','fused_softwarefp8'):
   fn=native if kind=='native' else fused
   sig=dict(base_sig)
   if kind!='native':sig.update(Q_OUT='*u8',K_CACHE='*u8',V_CACHE='*u8')
   const=dict(NUM_Q_HEADS=16,NUM_KV_HEADS=2,D=64,BD=64,BLOCK_M=block,HAS_ROPE=rope,HAS_KV_STORE=kind!='native',FP8_KV_STORE=kind!='native',USE_PDL=False,FP8_MAX=448.)
   if kind!='native':const['NATIVE']=kind=='fused_nativefp8'
   attrs={(i,):[['tt.divisibility',16]] for i in [0,1,2,3,4,5,6,7,9]}
   if block<32:attrs[(10,)]=[['tt.divisibility',16]]
   label=f'b{block}_rope{int(rope)}_{kind}'
   kernel=triton.compile(ASTSource(fn,signature=sig,constexprs=const,attrs=attrs),target=GPUTarget('cuda',90,32),options=dict(num_warps=4,debug=kind!='native'))
   ir=kernel.asm['ttgir'];ptx=kernel.asm['ptx']
   (out/(label+'.ttgir')).write_text(ir);(out/(label+'.ptx')).write_text(ptx)
   layouts=dict(re.findall(r'^(#\w+) = (#ttg\.blocked<[^\n]+>)$',ir,re.M))
   mul=next(line for line in ir.splitlines() if '%var = arith.mulf' in line)
   alias=re.search(r', (#\w+)>',mul).group(1)
   reductions=[line.strip() for line in ir.splitlines() if 'tt.reduce' in line or 'math.rsqrt' in line]
   item=dict(name=label,block=block,rope=rope,kind=kind,variance_layout=layouts[alias],reduction_lines=reductions,
     ir_sha256=hashlib.sha256(ir.encode()).hexdigest(),ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),metadata=kernel.metadata._asdict())
   records.append(item);print(label,item['variance_layout'],flush=True)
comparisons=[]
for block in (4,8,16,32):
 for rope in (False,True):
  rows={r['kind']:r for r in records if r['block']==block and r['rope']==rope}
  comparisons.append(dict(block=block,rope=rope,native_layout=rows['native']['variance_layout'],
    nativefp8_layout_equal=rows['native']['variance_layout']==rows['fused_nativefp8']['variance_layout'],
    softwarefp8_layout_equal=rows['native']['variance_layout']==rows['fused_softwarefp8']['variance_layout']))
receipt=dict(scope=__doc__,complete=True,compiler=triton.__version__,target='cuda90',cuda_devices_hidden=True,
 parent_source_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),candidate_source_sha256=hashlib.sha256((here/'prologue_package/kernel.py').read_bytes()).hexdigest(),
 elapsed_compile_seconds=time.monotonic()-started,comparisons=comparisons,records=records,
 default_runtime_max_m=2016,block32_default_runtime_reachable=False,
 limitation='Equal blocked layout is necessary evidence for the intended reduction, not proof of all instructions, floating behavior or deployed runtime specialization. No H200 execution.')
(out/'receipt.json').write_text(json.dumps(receipt,indent=2,default=str));print('complete',len(records),receipt['elapsed_compile_seconds'],flush=True)
