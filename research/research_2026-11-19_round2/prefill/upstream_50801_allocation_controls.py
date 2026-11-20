"""Execute pinned upstream op AST against an allocation recorder, not a GPU timing."""
import ast,json,hashlib
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Optional
HERE=Path(__file__).resolve().parent
SRC=HERE/'sources/sglang_50801/python/sglang/kernels/ops/moe/nvfp4_w4a16_sm90.py'
class Tensor:
 def __init__(self,shape,dtype='bf16',device='surrogate'):
  self.shape=tuple(shape) if not isinstance(shape,int) else (shape,);self.dtype=dtype;self.device=device
 @property
 def bytes(self):
  count=1
  for d in self.shape:count*=d
  return count*{'bf16':2,'int32':4,'int64':8,'fp32':4}[self.dtype]
records=[];alloc=[];calls=[]
def empty(shape,*,dtype,device):
 t=Tensor(shape,dtype,device);alloc.append(t);return t
ns=dict(torch=NS(Tensor=Tensor,empty=empty,bfloat16='bf16',int32='int32',int64='int64'),Optional=Optional,Nvfp4Sm90Experts=object,_VARIANTS={'auto':0,'single':1,'pair':2},_module=lambda _:NS(run=lambda *args:calls.append(args)))
node=next(n for n in ast.parse(SRC.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='nvfp4_w4a16_sm90_moe')
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SRC),'exec'),ns)
for t in (1,753,2048,4096,16384):
 for weighted in (False,True):
  for out_given in (False,True):
   alloc.clear();calls.clear();h=i=4096;e=32;k=8;r=t*k
   x=Tensor((t,h));ids=Tensor((t,k),'int32');w=Tensor((t,k),'fp32') if weighted else None
   supplied=Tensor((t if weighted else r,h)) if out_given else None
   experts=NS(gs13=Tensor((e,),'fp32'),gs2=Tensor((e,),'fp32'),intermediate=i,w13=None,w2=None)
   result=ns['nvfp4_w4a16_sm90_moe'](x,ids,experts,w,supplied)
   expected=4*(9*e+68+r//128)+8*r+2*r*h+2*r*i
   expected+=2*r*h if weighted or not out_given else 0
   expected+=2*t*h if weighted and not out_given else 0
   assert sum(a.bytes for a in alloc)==expected
   assert len(calls)==1
   if out_given:assert result is supplied
   assert result.shape==(t if weighted else r,h)
   common_raw=2*r*(max(2*i,h)+i)
   raw_core=2*r*(2*h+i)
   assert raw_core==common_raw
   records.append(dict(rows=t,weighted=weighted,out_given=out_given,allocated=expected,raw_core=raw_core,scored_humming_cache13_plus_cache2=common_raw,allocations=[dict(shape=a.shape,dtype=a.dtype,bytes=a.bytes) for a in alloc]))
doc=dict(passed=True,scope=__doc__,source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),cases=records,
 notes=['Raw route output required for Muse post-expert norm; generic weighted runner cannot be substituted.','Source-byte accounting excludes allocator reserved memory, native kernel internal scratch and captured graph lifetime.','No claim that all3.22GB at16K is extra: scored parent has same raw activation core.'])
(HERE/'sources/sglang_50801/allocation_controls.json').write_text(json.dumps(doc,indent=2))
print(json.dumps(dict(passed=True,cases=len(records))))
