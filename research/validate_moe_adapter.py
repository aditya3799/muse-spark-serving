"""Verify the patched Python MoE adapter preserves local routing contracts.
Marlin arithmetic is stubbed; actual alignment CUDA is exercised separately.
"""
import ast,contextlib,json,sys,types
from types import SimpleNamespace
from unittest.mock import patch
import torch
import validate_alignment as a
ROOT=a.ROOT
path=ROOT/'sglang/python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py'
tree=ast.parse(path.read_text(encoding='utf-8'))
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='MuseSpark1xFp4Experts')
fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_marlin_expert_outputs')
ns={'torch':torch,'_MARLIN_BLOCK_SIZE_M_CANDIDATES':(8,16,32,48,64)}; exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
method=ns['_marlin_expert_outputs']
a.FUNCS=a.compile_kernels(a.preamble+a.body,a.NAMES)
state={}; routes=[]
def map_ids(ids,mapping): return torch.where(ids<0,-1,mapping[ids.clamp_min(0)])
def generic(ids,block,experts,*,return_inactive_prefix_count):
 assert experts==state['local'] and not return_inactive_prefix_count
 state['path']='generic'; return a.align(ids,experts,block)
def single(ids,block,*,return_inactive_prefix_count):
 assert ids.shape[0]==1 and not return_inactive_prefix_count
 state['path']='single'; return a.align(ids,state['local'],block)
def gemm(x,c,*args,**kwargs):
 sorted_ids,experts,count=args[8:11]
 # Arguments follow the existing Marlin API: workspace follows the seven weight/metadata arguments.
 assert kwargs['is_ep'] and kwargs['use_fp32_reduce'] and not kwargs['mul_topk_weights']
 assert a.route_set((sorted_ids,experts,count),state['ids'],kwargs['moe_block_size'])==state['expected']
 c.fill_(float('nan'))
 for _,route in state['expected']: c[route].fill_(1)
 state['gemms']+=1
 return c
def silu(x,out,*,expert_ids,expert_step):
 assert expert_step==1 and torch.equal(expert_ids,state['ids'].view(-1))
 out.fill_(float('nan'))
 for _,route in state['expected']: out[route].fill_(1)
mods={}
exports={
 'sgl_kernel.scalar_type':{'scalar_types':SimpleNamespace(float4_e2m1f=object())},
 'sglang.kernels.ops.activation.activation':{'silu_and_mul':silu},
 'sglang.kernels.ops.moe.moe_wna16_marlin':{'moe_wna16_marlin_gemm':gemm},
 'sglang.srt.layers.moe.fused_moe_triton':{'moe_align_block_size':generic},
 'sglang.srt.layers.moe.token_dispatcher.standard':{'_map_topk_ids_to_local_experts':map_ids},
 'sglang.srt.layers.quantization.marlin_utils':{'marlin_make_workspace':lambda dev,**kw:torch.zeros(1,device=dev,dtype=torch.int32)},
 'sglang.kernels.ops.moe.moe_align_single_token':{'moe_align_single_token':single}}
for name,attrs in exports.items():
 parts=name.split('.')
 for i in range(1,len(parts)+1):
  key='.'.join(parts[:i])
  if key not in mods:
   mods[key]=types.ModuleType(key); mods[key].__path__=[]
 for k,v in attrs.items(): setattr(mods[name],k,v)
torch.manual_seed(37); cases=0
with patch.dict(sys.modules,mods):
 for local in [1,32,64]:
  for m in [1,2,7,16,32,128,513]:
   mapping=torch.full((256,),-1,device='cuda',dtype=torch.int32); offset=256-local
   mapping[offset:]=torch.arange(local,device='cuda',dtype=torch.int32)
   raw=torch.randint(-1,256,(m,8),device='cuda',dtype=torch.int32); raw[0,0]=offset
   ids=map_ids(raw,mapping)
   expected=sorted((int(e),j) for j,e in enumerate(ids.flatten().cpu().tolist()) if e>=0)
   state.update(local=local,ids=ids,expected=expected,gemms=0,path=None)
   obj=SimpleNamespace(intermediate_size_per_partition=64,num_local_experts=local,global_num_experts=256,_expert_map=mapping)
   for attr in ['w13_weight','w13_weight_scale','w13_weight_scale_2','w2_weight','w2_weight_scale','w2_weight_scale_2']: setattr(obj,attr,None)
   out,returned=method(obj,torch.randn(m,64,device='cuda',dtype=torch.bfloat16),torch.rand(m,8,device='cuda'),raw)
   assert torch.equal(returned,ids) and out.shape==(m,8,64)
   assert state['gemms']==2 and state['path']==('single' if m==1 else 'generic')
   for e,route in expected: assert torch.isfinite(out.view(-1,64)[route]).all()
   cases+=1
result={'cases':cases,'method':'actual patched MoE Python method; CUDA alignment; Marlin GEMMs and activation stubbed to verify table/shape/route wiring','full_marlin_arithmetic_tested':False}
(ROOT/'validation/moe_adapter.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

