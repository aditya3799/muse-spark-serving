"""Offline full-width Triton compiler report; never opens a CUDA context."""
import hashlib
import json
import os
import re
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]=""
import torch
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from finalize import _joined,_square,_shard_final
import importlib.util
here=Path(__file__).resolve().parent;root=here.parents[1]
frozen=root/"release_2026-11-18/candidate_sources/python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py"
spec=importlib.util.spec_from_file_location("frozen_finalize",frozen);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
cases=[(mod._post_expert_inv_rms_kernel,dict(x_ptr="*bf16",w_ptr="*bf16",ids_ptr="*i32",inv_ptr="*fp32",scale_ptr="*bf16",eps="fp32"),dict(HAS_SCALE=True,TOPK=8,H=4096,HAS_WEIGHT=True,WEIGHT_FIRST=True,BLOCK_H=4096),8)]
report=[]
for arch in (86,90):
    for fn,sig,cons,warps in cases:
        signature=dict(sig,**{k:"constexpr" for k in cons})
        attrs={(fn.arg_names.index(k),):[["tt.divisibility",16]] for k,v in sig.items() if v.startswith("*")}
        source=ASTSource(fn,signature,cons,attrs)
        compiled=triton.compile(source,target=GPUTarget("cuda",arch,32),options=dict(num_warps=warps,num_stages=2))
        ttgir=compiled.asm["ttgir"];ptx=compiled.asm["ptx"];cubin=compiled.asm["cubin"]
        row=dict(sm=arch,kernel=fn.__name__,threads=32*warps,shared_bytes=compiled.metadata.shared,
            cubin_sha256=hashlib.sha256(cubin).hexdigest(),cubin_bytes=len(cubin),
            blocked_layouts=[s.strip() for s in ttgir.splitlines() if s.startswith("#blocked")][:4],
            ptx_counts={k:len(re.findall(k,ptx)) for k in (r"shfl\.sync",r"bar\.sync",r"rsqrt\.approx",r"fma\.rn\.f32",r"add\.f32")})
        report.append(row)
        (here/"tiny_offline.json").write_text(json.dumps(report,indent=2)+"\n")
        print(arch,fn.__name__,row["shared_bytes"],flush=True)
assert not torch.cuda.is_initialized()
