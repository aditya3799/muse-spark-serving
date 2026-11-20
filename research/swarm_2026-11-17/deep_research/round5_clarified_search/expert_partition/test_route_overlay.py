import pathlib,json,ast,hashlib,random
import numpy as np
from replica_ownership_oracle import Session,Budget,analyze
from replica_route_overlay import map_core,PreparedReplicaRoutes
P=pathlib.Path(__file__).parent
rng=random.Random(3781);s=Session();train=[list(range(8))for _ in range(128)];s.observe(train);plan=s.choose(Budget(8),max_copies=8)
owners=np.full((256,8),-1,np.int32);counts=np.empty(256,np.int32)
for e,hs in enumerate(plan.holders):owners[e,:len(hs)]=hs;counts[e]=len(hs)
checks=0
for m in [0,1,8,9,32,129]:
    for gen in range(8):
        rows=[rng.sample(range(256),8)for _ in range(m)]
        if gen%2 and m:rows[-1][-3:]=[-1]*3
        if gen==0:rows=[list(range(8))for _ in range(m)]
        ids=np.array(rows,np.int32).reshape(m,8);expected=s.route(rows,plan,'token_mod')
        got=[]
        for rank in range(8):
            slots=np.array([v if v is not None else -1 for v in plan.slots[rank]],np.int32)
            result=map_core(ids,owners,counts,slots,rank,np.arange(m,dtype=np.int64)[:,None],np.where)
            assert result.dtype==np.int32
            assert np.array_equal(result,np.array(expected[rank],np.int32).reshape(m,8))
            got.append(result.tolist());checks+=1
        analyze(rows,plan,got)
manifest=json.loads((P/'ROUTE_OVERLAY_MANIFEST.json').read_text());base=pathlib.Path(manifest['baseline_source']).read_text(encoding='utf8');new=(P/'muse_spark_fp4_moe_replica_overlay.py').read_text(encoding='utf8')
def mapping_body(txt):
    tree=ast.parse(txt)
    fn=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='_marlin_expert_outputs')
    humming=next(n for n in fn.body if isinstance(n,ast.If)and ast.unparse(n.test)=='self.use_humming')
    return humming.body
old=mapping_body(base);cur=mapping_body(new)
oldlocal=old[1:3]
branch=next(n for n in cur if isinstance(n,ast.If)and ast.unparse(n.test)=='replica_routes is None')
assert ast.dump(ast.Module(body=oldlocal,type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=branch.body,type_ignores=[]),include_attributes=False)
try:PreparedReplicaRoutes(layer=None,owners=None,counts=None,slots=None,row_numbers=None,rank=0,torch_module=None)
except RuntimeError:held=True
else:raise AssertionError('constructor must not admit incomplete storage')
report={'status':'PASS_CPU_NUMPY_MAP_AND_DEFAULT_BRANCH_AST','map_comparisons':checks,'changed_shapes_and_routes':48,'default_off_original_AST_identical':True,'constructor_held':held,'limitations':['Numpy evaluates indexing formula, not CUDA execution or capture.','No weight capacity or rank-coherent publication implementation.','Current token-mod mapping can imbalance strided expert occurrences.'],'hashes':{n:hashlib.sha256((P/n).read_bytes()).hexdigest()for n in ['replica_route_overlay.py','replica_route_overlay.diff','test_route_overlay.py']}}
(P/'ROUTE_OVERLAY_CPU_RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
