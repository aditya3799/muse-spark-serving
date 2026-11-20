import ast,json,pathlib,hashlib
from copy import deepcopy
from types import SimpleNamespace as N
P=pathlib.Path(__file__).resolve().parent;R=next(p for p in P.parents if (p/'RULES.md').exists())
manifest=json.loads((P/'CAPACITY_OVERLAY_MANIFEST.json').read_text());source=pathlib.Path(manifest['source'])
prior=R/'swarm_2026-11-17/deep_research/round3_structural/expert_partition/SWAP_FULL_HELPER_PLAN.json'
ref=json.loads(prior.read_text())['reference']
class T:
 def __init__(self,shape,dtype='i32',values=None):self.shape=shape;self.ndim=len(shape);self.dtype=dtype;self.device=0;self.is_cuda=True;self.values=values
 def numel(self):
  n=1
  for v in self.shape:n*=v
  return n
 def is_contiguous(self):return True

def partition(ids,experts,cutoff):
 vals=ids.values;counts=[sum(v==e for v in vals)for e in range(experts)]
 light=[v if 0<=v<experts and counts[v]<=cutoff else -1 for v in vals]
 heavy=[v if 0<=v<experts and counts[v]>cutoff else -1 for v in vals]
 for v,a,b in zip(vals,light,heavy):assert ((a>=0)+(b>=0))==int(0<=v<experts)
 return dict(physical_experts=experts,cutoff=cutoff,counts=counts,last_slot_count=counts[-1],invalid_preserved=all(a==b==-1 for v,a,b in zip(vals,light,heavy)if not 0<=v<experts))

def extracted(path):
 tree=ast.parse(path.read_text(encoding='utf8'));nodes=[]
 for name in ['select_ep_config','choose_partition_configs','split_expert_outputs']:
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
  if name=='split_expert_outputs':
   kept=[]
   for stmt in fn.body:
    if isinstance(stmt,(ast.Import,ast.ImportFrom)):continue
    if isinstance(stmt,ast.Assign) and any(isinstance(t,ast.Tuple) for t in stmt.targets):
     # Stop at actual physical histogram boundary, before any GPU operation.
     assert ast.unparse(stmt.targets[0])=='(light_ids, heavy_ids, counts)'
     kept.append(ast.Return(value=ast.Call(func=ast.Name(id='partition_routes',ctx=ast.Load()),args=[ast.Name(id='local_ids',ctx=ast.Load()),ast.Name(id='experts',ctx=ast.Load()),ast.Subscript(value=ast.Name(id='plan',ctx=ast.Load()),slice=ast.Constant('cutoff'),ctx=ast.Load())],keywords=[])))
     break
    kept.append(stmt)
   fn.body=kept
  nodes.append(fn)
 mod=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]));ns={'deepcopy':deepcopy,'partition_routes':partition};exec(compile(mod,str(path),'exec'),ns)
 return ns['split_expert_outputs']
old=extracted(source);new=extracted(P/'humming_split_capacity_overlay.py');torch=N(bfloat16='bf16',int32='i32',int64='i64')
def layer(cap,policy='absent',w2=None):
 # Boundary fixture: authentic previously-qualified light/heavy config objects,
 # but controlled [0,32]/[32,4096] intervals, not claimed actual entire tuner table.
 x=N(w13_post_scale=T((cap,)),w2_post_scale=T((cap if w2 is None else w2,)),fp32_configs={st:[(0,32,deepcopy(ref['light'][st])),(32,4096,deepcopy(ref['heavy'][st]))]for st in ['w13','w2']})
 if policy!='absent':x._research_policy_local_experts=policy
 return x

def run(fn,cap,policy='absent',global_e=256,w2=None):
 vals=([0,cap-1,-1,cap,-2,1,2,3]*32);ids=T((32,8),values=vals)
 return fn(lambda *a,**k:'WHOLE_HELPER_FALLBACK',layer(cap,policy,w2),T((32,4096),'bf16'),ids,torch,global_experts=global_e)
results=[]
for cap in [32,33,40,64]:
 a,b=run(old,cap),run(new,cap);assert a==b
 c=run(new,cap,32);assert isinstance(c,dict)and c['physical_experts']==cap and c['last_slot_count']==32 and c['invalid_preserved']
 results.append(dict(capacity=cap,original=a,default_off=b,opt_in_policy32=c))
neg=[]
for policy in [0,-1,True,None,33,257]:
 if policy==33:cap=40
 else:cap=33
 assert run(new,cap,policy)=='WHOLE_HELPER_FALLBACK';neg.append(['policy',policy])
assert run(new,33,32,w2=32)=='WHOLE_HELPER_FALLBACK';neg.append('scale_count_mismatch')
assert run(new,33,32,global_e=32)=='WHOLE_HELPER_FALLBACK';neg.append('global_less_than_capacity')
report={'status':'PASS_ACTUAL_HELPER_DISPATCH_PREFIX_CPU','cases':results,'negatives':neg,'scope':'Executed actual source-extracted helper validation/config dispatch through histogram boundary; CPU histogram sentinel oracle. Authentic light/heavy config objects, controlled interval fixture. No GEMM, native table regeneration, tensor packing, GPU or serving claim.','key_finding':'Original physical33/40 causes whole-helper fallback due256%capacity; opt-in policy32 reaches split while physical histograms retain33/40/64slots. Absent attribute decisions equal original.','hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in [source,P/'humming_split_capacity_overlay.py',prior,P/'test_capacity_dispatch.py']}}
(P/'CAPACITY_DISPATCH_CPU_RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'capacities':[32,33,40,64],'negative_count':len(neg)}))
