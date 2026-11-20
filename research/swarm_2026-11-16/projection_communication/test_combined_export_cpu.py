"""CPU-only exact-source metadata/fallback and actual caller-forwarding checks."""
import ast,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
P=Path(__file__).resolve().parent;ROOT=P/'adaptive_split_v3/serving_export';BASE=P.parent/'moe/candidate_v3/serving_export'
SPLIT=ROOT/'python/sglang/kernels/ops/moe/muse_spark_humming_split.py'
NV=ROOT/'python/sglang/srt/layers/muse_spark/muse_spark_nvfp4_moe.py'
FP=ROOT/'python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
class NoImports(ast.NodeTransformer):
 def visit_Import(self,node):return None
 def visit_ImportFrom(self,node):return None

def fn(path,name):
 return next(n for n in ast.walk(ast.parse(path.read_text(encoding='utf-8'))) if isinstance(n,ast.FunctionDef) and n.name==name)
def load(functions,ns):
 tree=ast.Module(body=[NoImports().visit(copy.deepcopy(f)) for f in functions],type_ignores=[]);ast.fix_missing_locations(tree);exec(compile(tree,'<exact-source-no-import-metadata-test>','exec'),ns)
class Tensor:
 def __init__(self,shape,dtype='bf16',device='cuda',contig=True):self.shape=shape;self.ndim=len(shape);self.dtype=dtype;self.device=device;self.is_cuda=device=='cuda';self.contig=contig
 def numel(self):
  n=1
  for v in self.shape:n*=v
  return n
 def is_contiguous(self):return self.contig
 def contiguous(self):return self
 def to(self,dtype):return self
 def clamp_min(self,value):return self
 def __add__(self,value):return self
class Lookup:
 def __getitem__(self,key):return key
T=SimpleNamespace(bfloat16='bf16',int32='i32',int64='i64');calls=[]
def fallback(*args,**kwargs):calls.append((args,kwargs));return 'fallback'
def plan_none(*args):return None
ns={'torch':T,'choose_partition_configs':plan_none}
load([fn(SPLIT,'split_expert_outputs')],ns)
base=SimpleNamespace(w13_post_scale=Tensor((32,)),w2_post_scale=Tensor((32,)))
count=0
for defer in (False,True):
 for retain in (False,True):
  for kind in ('absent_global','uneven_global','smaller_global','zero_global','bool_global','route_limit','zero_routes','cpu_x','bad_dtype','bad_ids_dtype','noncontiguous','wrong_rank','row_mismatch','same_pair'):
   x=Tensor((128,4096));ids=Tensor((128,8),'i32');global_e=256
   if kind=='absent_global':global_e=None
   elif kind=='uneven_global':global_e=257
   elif kind=='smaller_global':global_e=16
   elif kind=='zero_global':global_e=0
   elif kind=='bool_global':global_e=True
   elif kind=='route_limit':x=Tensor((513,4096));ids=Tensor((513,8),'i32')
   elif kind=='zero_routes':x=Tensor((0,4096));ids=Tensor((0,8),'i32')
   elif kind=='cpu_x':x=Tensor((128,4096),device='cpu')
   elif kind=='bad_dtype':x=Tensor((128,4096),dtype='f32')
   elif kind=='bad_ids_dtype':ids=Tensor((128,8),'f32')
   elif kind=='noncontiguous':ids=Tensor((128,8),'i32',contig=False)
   elif kind=='wrong_rank':ids=Tensor((1024,),'i32')
   elif kind=='row_mismatch':ids=Tensor((127,8),'i32')
   assert ns['split_expert_outputs'](fallback,base,x,ids,T,global_e,retain=retain,defer_downscale=defer)=='fallback'
   assert calls[-1][1]=={'retain':retain,'defer_downscale':defer};count+=1
# Public wrapper passes private fallback callback, never itself.
wrapped=[]
def split_spy(*args,**kwargs):wrapped.append((args,kwargs));return 'wrapped'
w={'split_expert_outputs':split_spy,'_humming_expert_outputs_default':fallback};load([fn(NV,'_humming_expert_outputs')],w)
for defer in (False,True):
 for global_e in (None,256):
  assert w['_humming_expert_outputs'](base,Tensor((128,4096)),Tensor((128,8),'i32'),T,defer_downscale=defer,global_experts=global_e)=='wrapped'
  assert wrapped[-1][0][0] is fallback and wrapped[-1][0][-1]==global_e and wrapped[-1][1]['defer_downscale']==defer;count+=1
# Actual class method source forwards outer metadata and flags, with/without expert map.
forwarded=[]
def helper_spy(*args,**kwargs):forwarded.append((args,kwargs));return 'forwarded'
fns={'_humming_expert_outputs':helper_spy,'torch':T,'torch_module':T,'torchTensor':Tensor}
method=fn(FP,'_marlin_expert_outputs');method.returns=None
for arg in method.args.args+method.args.kwonlyargs:arg.annotation=None
load([method],fns)
for global_e in (32,256,512):
 for defer in (False,True):
  for mapped in (False,True):
   outer=SimpleNamespace(use_humming=True,_expert_map=object() if mapped else None,_expert_lookup=Lookup(),_humming_layer=base,global_num_experts=global_e)
   assert fns['_marlin_expert_outputs'](outer,Tensor((128,4096)),None,Tensor((128,8),'i32'),defer_downscale=defer)=='forwarded'
   assert forwarded[-1][1]=={'defer_downscale':defer,'global_experts':global_e};count+=1
# Parent AST differs only in the added global_experts keyword at the actual call.
original=ast.parse((BASE/FP.relative_to(ROOT)).read_text(encoding='utf-8'));changed=ast.parse(FP.read_text(encoding='utf-8'))
removed=0
for node in ast.walk(changed):
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='_humming_expert_outputs':
  before=len(node.keywords);node.keywords=[k for k in node.keywords if k.arg!='global_experts'];removed+=before-len(node.keywords)
assert removed==1 and ast.dump(original)==ast.dump(changed)
report={'status':'PASS_CPU_EXACT_SOURCE_DISPATCH','cases':count,'scope':'Metadata/control-flow tests using exact function AST with imports removed and tensor metadata stubs; no arithmetic/CUDA proof','actual_callsite_forwards_global_metadata':True,'parent_ast_only_added_global_keyword':True,'fallback_recursion_absent':True,'defer_and_retain_preserved':True,'hashes':{f.relative_to(ROOT).as_posix():sha(f) for f in (SPLIT,NV,FP)}}
(P/'adaptive_split_v3/cpu_dispatch_review.json').write_bytes((json.dumps(report,indent=2)+'\n').encode());print(json.dumps(report))
