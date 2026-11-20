"""Compile the dynamic live-expert native restorer at production shapes for both GPUs, CPU only."""
import hashlib
import json
import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]=""
import torch
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from exclusive_pack import _restore_native,_present,_compact

here=Path(__file__).resolve().parent
out=[]
for arch in (86,90):
    for fn,ptr,cons in [
        (_present,dict(ids="*i32",flags="*i32"),dict(S=256*8,E=32,B=256)),
        (_compact,dict(flags="*i32",active="*i32"),dict(E=32,B=32)),
        *((_restore_native,dict(q="*u8",sc="*i32",active="*i32",words="*i32",bytes_out="*u8"),
           dict(E=32,N=n,K=4096,WG=wg,WORKERS=264 if arch==90 else 40))
           for n in (4096,8192) for wg in (False,True))]:
        signature=dict(ptr,**{k:"constexpr" for k in cons})
        attrs={(fn.arg_names.index(k),):[["tt.divisibility",16]] for k in ptr}
        src=ASTSource(fn,signature,cons,attrs)
        c=triton.compile(src,target=GPUTarget("cuda",arch,32),options=dict(num_warps=4))
        cubin=c.asm["cubin"]
        out.append(dict(architecture=arch,function=fn.__name__,constants=cons,bytes=len(cubin),
                        shared=c.metadata.shared,sha256=hashlib.sha256(cubin).hexdigest()))
assert not torch.cuda.is_initialized()
(here/"exclusive_compile.json").write_text(json.dumps(out,indent=2))
print("Compiled",len(out),"live-route native fallback variants")
