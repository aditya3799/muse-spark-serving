"""Compile the exact pinned parent's selector kernel for an equally fused local comparison."""
from pathlib import Path
import hashlib
import json
import re
import sys
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parents[2]/"research_2026-11-18_window_repair/decode_selector"))
from original_walk import _selector_walk_kernel as fn
types=["*bf16","*i64","*fp32","*fp32","*i32","*i64","*fp32"]
args=dict(zip(fn.arg_names,types))
cons=dict(slots=7,top_k=16)
source=ASTSource(fn,dict(args,**{k:"constexpr" for k in cons}),cons,
                 {(i,):[["tt.divisibility",16]] for i in range(7)})
obj=triton.compile(source,target=GPUTarget("cuda",86,32),options=dict(num_warps=1))
entry=re.search(r"\.entry\s+(\w+)\s*\((.*?)\)",obj.asm["ptx"],re.S)
params=re.findall(r"\.param\s+\.([a-z0-9]+)",entry.group(2))
assert len(params)>=7 and set(params)=={"u64"},params
binary=obj.asm["cubin"]
(here/"chain_sm86.cubin").write_bytes(binary)
row=dict(symbol=entry.group(1),params=params,shared=obj.metadata.shared,threads=32,
         sha256=hashlib.sha256(binary).hexdigest(),sm=86,
         global_scratch_size=getattr(obj.metadata,"global_scratch_size",0),
         profile_scratch_size=getattr(obj.metadata,"profile_scratch_size",0))
assert row["global_scratch_size"]==row["profile_scratch_size"]==0
(here/"chain_abi.json").write_text(json.dumps(row,indent=2))
print(row)
