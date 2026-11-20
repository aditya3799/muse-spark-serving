import ast,copy,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
S=P.parents[1]/'coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang'
candidate=ast.parse((P/'scale_activation_helper.py').read_text())
sources={'split_expert_outputs':S/'kernels/ops/moe/muse_spark_humming_split.py','_humming_expert_outputs_default':S/'srt/layers/muse_spark/muse_spark_nvfp4_moe.py'}
rows=[]
for name,path in sources.items():
    expected=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    actual=copy.deepcopy(next(n for n in candidate.body if isinstance(n,ast.FunctionDef) and n.name==name))
    matches=0
    for i,node in enumerate(actual.body):
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='scaled_silu_and_mul':
            assert isinstance(actual.body[i-1],ast.Assign) and actual.body[i-1].targets[0].id=='cache2'
            alloc=actual.body[i-1]
            reference=ast.parse('scale_active_rows(cache1,local_ids,layer.w13_post_scale)\nsilu_and_mul(cache1,cache2,expert_ids=local_ids.flatten(),expert_step=1)').body
            actual.body[i-1:i+1]=[reference[0],alloc,reference[1]]
            matches+=1;break
    assert matches==1 and ast.dump(actual)==ast.dump(expected),name
    rows.append({'function':name,'ast_equal_after_reversing_only_scale_activation_fusion':True,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
original=next(n for n in ast.walk(ast.parse((P/'grouped_actual_probe.py').read_text())) if isinstance(n,ast.FunctionDef) and n.name=='check')
new=next(n for n in ast.walk(ast.parse((P/'scale_activation_fullchain_probe.py').read_text())) if isinstance(n,ast.FunctionDef) and n.name=='check')
assert ast.dump(original)==ast.dump(new)
report={'status':'PASS_SOURCE_AST_EQUIVALENCE','functions':rows,'fullchain_check_and_failure_journal_unchanged':True,'candidate_sha256':hashlib.sha256((P/'scale_activation_helper.py').read_bytes()).hexdigest(),'scope':'Static proof of unchanged GEMM/config/partition/helper control flow except exact scale+activation replacement; not GPU equivalence'}
(P/'scale_activation_source_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))