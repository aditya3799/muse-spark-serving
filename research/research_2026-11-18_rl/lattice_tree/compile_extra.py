"""Compiler-only verification for research kernels on SM86 and SM90."""
import hashlib
import json
import os
import sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]=""
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
import triton
here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parent/"expert_slab"))
from slab import _decode
from linear_planner import _linear
from tree_attention import _part,_finish

tasks=[(_decode,["*i32","*u8","*bf16","*i32","*bf16"],dict(N=8192,K=4096,TRANSPOSE_MINI=False,BN=16,BK=64)),
       (_linear,["*fp32","*i64","*i64","fp32","fp32"],dict(SLOTS=7,K=16)),
       (_finish,["*fp32"]*4,dict(SPLITS=37,NS=64,M=64,N=8,G=8,H=2,D=64))]
for left in (0,2048):
    for tree in (False,True):
        tasks.append((_part,["*u8","*u8","*u8","*i32","*i32","*i32","*i32","*u32",
                              "*i32","*fp32","*fp32","*fp32","*fp32","i32","fp32"],
                       dict(SPLITS=37 if not left else 1,CHUNK=2048,LEFT=left,N=8,G=8,H=2,D=64,M=64,BK=32,TREE=tree,DIRECT=False)))
out=[]
for sm in (86,90):
    for kernel,types,constants in tasks:
        args=dict(zip(kernel.arg_names,types))
        attrs={(i,):[["tt.divisibility",16]] for i,v in enumerate(types) if v.startswith("*")}
        src=ASTSource(kernel,dict(args,**{k:"constexpr" for k in constants}),constants,attrs)
        compiled=triton.compile(src,target=GPUTarget("cuda",sm,32),options=dict(num_warps=4,num_stages=2))
        binary=compiled.asm["cubin"]
        out.append(dict(sm=sm,kernel=kernel.__name__,constants=constants,shared_bytes=compiled.metadata.shared,
                        cubin_bytes=len(binary),sha256=hashlib.sha256(binary).hexdigest()))
        print(sm,kernel.__name__,constants.get("LEFT"),constants.get("TREE"),len(binary),flush=True)
(here/"extra_codegen.json").write_text(json.dumps(out,indent=2))
