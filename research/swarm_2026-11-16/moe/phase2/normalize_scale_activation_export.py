"""Canonical UTF-8/LF production export; preserve every executed source."""
import ast,hashlib,json,re
from pathlib import Path
P=Path(__file__).resolve().parent;OUT=P/'scale_activation_export'
OLD=OUT/'full_candidate_sources_clean';NEW=OUT/'full_candidate_sources_utf8_lf'
BASE=P.parents[1]/'coordination/round5_humming_adaptive_fused/candidate_sources'
assert not NEW.exists(),'Use a fresh normalized output'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prior=json.loads((OUT/'full_snapshot_manifest.json').read_bytes().decode('utf-8'))
pattern=r'(?m)^    scale_active_rows\(cache1,\s*local_ids,\s*layer.w13_post_scale\)\n(    cache2 = [^\n]+)\n    silu_and_mul\(cache1,\s*cache2,\s*expert_ids=local_ids.flatten\(\),\s*expert_step=1\)'
modified={'python/sglang/srt/layers/muse_spark/muse_spark_nvfp4_moe.py':'_humming_expert_outputs_default','python/sglang/kernels/ops/moe/muse_spark_humming_split.py':'split_expert_outputs'}
translations=[]
for rel,digest in prior['sources'].items():
    path=OLD/rel;assert sha(path)==digest
    old_text=path.read_bytes().decode('utf-8').replace('\r\n','\n').replace('\r','\n')
    text=old_text
    if rel in modified:
        original=(BASE/rel).read_bytes().decode('utf-8').replace('\r\n','\n').replace('\r','\n')
        text,count=re.subn(pattern,r'\1\n    from sglang.kernels.ops.moe.muse_spark_scale_activation import scaled_silu_and_mul\n    scaled_silu_and_mul(cache1,cache2,local_ids,layer.w13_post_scale)',original)
        assert count==1
    if path.suffix=='.py':assert ast.dump(ast.parse(old_text))==ast.dump(ast.parse(text)),rel
    else:assert text==old_text,rel
    dest=NEW/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(text.encode('utf-8'))
    assert b'\r' not in dest.read_bytes()
    translations.append({'path':rel,'executed_snapshot_sha256':digest,'utf8_lf_sha256':sha(dest),'newline_only':text==old_text,'python_ast_identical':path.suffix=='.py','decoded_as':'utf-8'})
checks=[]
for rel,name in modified.items():
    original=ast.parse((BASE/rel).read_bytes().decode('utf-8'));changed=ast.parse((NEW/rel).read_bytes().decode('utf-8'))
    fn=next(n for n in ast.walk(changed) if isinstance(n,ast.FunctionDef) and n.name==name)
    fn.body=[n for n in fn.body if not(isinstance(n,ast.ImportFrom) and n.module=='sglang.kernels.ops.moe.muse_spark_scale_activation')]
    count=0
    for i,n in enumerate(fn.body):
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='scaled_silu_and_mul':
            alloc=fn.body[i-1];assert isinstance(alloc,ast.Assign) and alloc.targets[0].id=='cache2'
            ref=ast.parse('scale_active_rows(cache1,local_ids,layer.w13_post_scale)\nsilu_and_mul(cache1,cache2,expert_ids=local_ids.flatten(),expert_step=1)').body
            fn.body[i-1:i+1]=[ref[0],alloc,ref[1]];count+=1;break
    assert count==1 and ast.dump(changed)==ast.dump(original),rel
    checks.append({'file':rel,'utf8_full_file_ast_equals_scored_baseline_after_reversing_only_fusion':True})
report={**prior,'status':'EXPERIMENTAL_UTF8_LF_SNAPSHOT_LOCAL_GATES_PASS_NO_PERFORMANCE_PROMOTION','source_root':str(NEW),'sources':{row['path']:row['utf8_lf_sha256'] for row in translations},'hash_translation':translations,'structural_checks':checks,'original_executed_snapshot_preserved':True,'validation_scope':'UTF-8 decode plus LF normalization; all Python ASTs identical to exact-executed export, header text identical. No GPU rerun necessary; earlier runtime and SM90 evidence transfers only through this source-equivalence mapping.'}
report['overlay_source_hashes']={rel:report['sources'][rel] for rel in prior['overlay_source_hashes']}
manifest=OUT/'full_snapshot_manifest_utf8_lf.json';manifest.write_bytes((json.dumps(report,indent=2)+'\n').encode('utf-8'))
print(json.dumps({'source_root':str(NEW),'files':len(translations),'all_newline_only':all(row['newline_only'] for row in translations),'manifest_sha256':sha(manifest),'original_manifest_unchanged_sha256':sha(OUT/'full_snapshot_manifest.json')}))