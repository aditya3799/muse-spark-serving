"""Export our own Triton SM86 selector for the Windows PyTorch run using CUDA's driver ABI."""
import hashlib
import json
import re
import sys
from pathlib import Path
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
here = Path(__file__).resolve().parent
sys.path.insert(0,str(here.parent))
from fused_tree import _grow

args=dict(ids="*i64", scores="*fp32", log_mass="*fp32", anchor="*i64", out_ids="*i64",
          out_parents="*i32", out_pos="*i32", out_mask="*u32", inv_temp="fp32", log_fork="fp32")
width=int(sys.argv[1]) if len(sys.argv)>1 else 8
assert width in (8,12,16)
kind="planner" if width==8 else f"planner{width}"
cons=dict(DEPTH=7,K=16,N=width,ROWS=triton.next_power_of_2(width),MASS=True)
attrs={(i,):[["tt.divisibility",16]] for i,v in enumerate(args.values()) if v.startswith("*")}
src=ASTSource(_grow,dict(args,**{k:"constexpr" for k in cons}),cons,attrs)
obj=triton.compile(src,target=GPUTarget("cuda",86,32),options=dict(num_warps=4,num_stages=2))
ptx=obj.asm["ptx"]
entry=re.search(r"\.entry\s+(\w+)\s*\((.*?)\)",ptx,re.S)
assert entry
params=re.findall(r"\.param\s+\.([a-z0-9]+)",entry.group(2))
assert params[:10]==["u64"]*8+["f32"]*2 and all(p=="u64" for p in params[10:]), params
binary=obj.asm["cubin"]
(here/(kind+"_sm86.cubin")).write_bytes(binary)
row=dict(nodes=width,symbol=entry.group(1),params=params,shared=obj.metadata.shared,threads=obj.metadata.num_warps*32,
         sha256=hashlib.sha256(binary).hexdigest(),sm=86,
         global_scratch_size=getattr(obj.metadata,"global_scratch_size",0),
         profile_scratch_size=getattr(obj.metadata,"profile_scratch_size",0))
assert row["global_scratch_size"]==row["profile_scratch_size"]==0
(here/(kind+"_abi.json")).write_text(json.dumps(row,indent=2))
print(row)
