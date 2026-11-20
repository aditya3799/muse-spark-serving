from pathlib import Path
import ast,hashlib,json,sys
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
from virtual_k_gate_contract import require_h200,route_arrays
require_h200(9,0,'NVIDIA H200')
negative=0
for args in [(8,6,'RTX 3050'),(9,0,'H100'),(10,0,'B200'),(9,1,'H200')]:
 try:require_h200(*args)
 except RuntimeError:negative+=1
 else:raise AssertionError(args)
cases=0
for m in (1,8,14,15,32,128,256):
 for bm in (8,16,24,32,64):
  for pattern in ('uniform','hot_expert','mixed_duplicate','all_remote'):
   r=route_arrays(m,32,bm,pattern,5);seen=[]
   assert r['padded']%bm==0
   for start in range(0,r['padded'],bm):
    e=r['expert_ids'][start//bm]
    for index in r['sorted_ids'][start:start+bm]:
     if index<m*8:assert r['local_ids'][index]==e;seen.append(index)
   assert sorted(seen)==[i for i,x in enumerate(r['active']) if x]
   assert len(seen)==len(set(seen))
   assert all(x==m*8 for x in r['sorted_ids'][r['padded']:])
   assert all(x==-1 for x in r['expert_ids'][r['padded']//bm:])
   cases+=1
source=(P/'virtual_k_gate_sm90.py').read_text(encoding='utf-8-sig');tree=ast.parse(source)
main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
guard=next(x.lineno for x in ast.walk(main) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='require_h200')
humming=next(x.lineno for x in main.body if isinstance(x,ast.ImportFrom) and x.module=='humming')
assert guard<humming
assert 'super().prepare()' in source and 'ops.launch_kernel(configs=tables[i]' in source
assert 'HummingKernel.prepare_kernels' not in source
assert "j['configuration']!=j['baseline']" in source
r={'status':'CPU_contract_passed','route_geometries':cases,'device_rejections':negative,'architecture_guard_precedes_launcher_import':True,'harness_sha256':hashlib.sha256((P/'virtual_k_gate_sm90.py').read_bytes()).hexdigest(),'limits':'No imports of Torch/Humming, no CUDA context, no compiler or device execution. Runtime API compatibility and numerical gates remain unexecuted on H200.'}
(P/'VIRTUAL_K_GATE_CPU_AUDIT.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r))
