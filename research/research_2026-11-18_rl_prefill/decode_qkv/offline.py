"""Compile the exact frozen entry for Hopper using Triton's offline NVIDIA target."""
import ast,hashlib,json,sys,types
from pathlib import Path
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
H=Path(__file__).resolve().parent
src=H/'muse_spark_qk_norm_rope_store.py'
fn=next(n for n in ast.parse(src.read_text()).body if getattr(n,'name',None)=='_qk_rms_norm_fwd_kernel')
mod=types.ModuleType('qkv_offline_frozen');mod.__file__=str(src);mod.__dict__.update(tl=tl,triton=triton,FP8_E4M3_MAX=448.0);sys.modules[mod.__name__]=mod
exec(compile(ast.Module([fn],[]),str(src),'exec'),mod.__dict__)
f=mod._qk_rms_norm_fwd_kernel
sigs=['*bf16','*u8','*bf16','*i64','*fp32','*u8','*u8','*i64','*u8','fp32','i32','i32','i32']
compiled=[]
for num_rows,has_rope in ((160,False),(160,True),(5120,True)):
    block_m=4 if num_rows<1024 else 8
    cfg=dict(NUM_Q_HEADS=16,NUM_KV_HEADS=2,D=64,BD=64,BLOCK_M=block_m,HAS_ROPE=has_rope,HAS_KV_STORE=True,FP8_KV_STORE=True,FA3_BYTE_STORE=True,USE_PDL=True,FP8_MAX=448.)
    sig=dict(zip(f.arg_names,sigs));sig.update({k:'constexpr' for k in cfg})
    binary=triton.compile(ASTSource(f,signature=sig,constexprs=cfg),target=GPUTarget('cuda',90,32),options={'num_warps':4})
    ptx=binary.asm['ptx']; cubin=binary.asm['cubin']
    assert ('griddepcontrol.' in ptx or 'gdc.' in ptx) and 'trap;' in ptx, [x for x in ptx.splitlines() if 'grid' in x or 'trap' in x]
    compiled.append(dict(rows_hint=num_rows,rope=has_rope,block_m=block_m,ptx_target=next(l for l in ptx.splitlines() if l.startswith('.target')),ptx_has_pdl=True,ptx_has_bounds_trap=True,cubin_bytes=len(cubin),cubin_hash=hashlib.sha256(cubin).hexdigest()))
print(json.dumps(compiled,indent=2));(H/'offline_sm90.json').write_text(json.dumps(compiled,indent=2))

