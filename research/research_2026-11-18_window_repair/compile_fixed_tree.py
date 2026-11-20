import hashlib,json,os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]=""
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
import triton
from tree_compact_pages import _compact_swa
H=Path(__file__).resolve().parent
r=[]
for sm in (86,90):
    for mt in ("*i32","*i64"):
        types=["*i32","*i32","*i32","*i32","*i32",mt,"*i32","i32","i32","i32"]
        c=dict(LEFT=2047,WIDTH=8,BLOCK=256)
        sig=dict(zip(_compact_swa.arg_names,types));sig.update({k:"constexpr" for k in c})
        src=ASTSource(_compact_swa,sig,c)
        mod=triton.compile(src,target=GPUTarget("cuda",sm,32),options={"num_warps":4})
        image=mod.asm["cubin"]
        r.append(dict(sm=sm,dtype=mt,bytes=len(image),shared_bytes=mod.metadata.shared,
                      sha256=hashlib.sha256(image).hexdigest()))
(H/"fixed_tree/compact_codegen.json").write_text(json.dumps(r,indent=2));print(r)

