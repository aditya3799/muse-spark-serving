import ast,copy,hashlib,json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;OUT=P/'scale_activation_export'
BASE=P.parents[1]/'coordination/round5_humming_adaptive_fused/candidate_sources';OVER=OUT/'candidate_sources';DEST=OUT/'full_candidate_sources_clean'
# Reuse only the interrupted snapshot after validating every existing byte.
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(BASE.parent/'candidate.diff')=='8b3ed81e053e4ccb2a043afd25cb94bb49017995ece517bb35aeb6e1b79e026b'
assert not DEST.exists(),'Use a fresh clean source destination'
baseline_manifest=json.loads((BASE.parent/'package_validation.json').read_text(encoding='utf-8'))
for relative,digest in baseline_manifest['source_hashes'].items():
    source=BASE/relative;assert sha(source)==digest
    target=DEST/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
for rel,digest in manifest['sources'].items():
    assert sha(OVER/rel)==digest
    target=DEST/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OVER/rel,target)
checks=[]
for rel,name in (('python/sglang/srt/layers/muse_spark/muse_spark_nvfp4_moe.py','_humming_expert_outputs_default'),('python/sglang/kernels/ops/moe/muse_spark_humming_split.py','split_expert_outputs')):
    original=ast.parse((BASE/rel).read_text(encoding='utf-8'));changed=ast.parse((DEST/rel).read_text(encoding='utf-8'))
    fn=next(n for n in ast.walk(changed) if isinstance(n,ast.FunctionDef) and n.name==name)
    fn.body=[n for n in fn.body if not(isinstance(n,ast.ImportFrom) and n.module=='sglang.kernels.ops.moe.muse_spark_scale_activation')]
    count=0
    for i,n in enumerate(fn.body):
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='scaled_silu_and_mul':
            alloc=fn.body[i-1];assert isinstance(alloc,ast.Assign) and alloc.targets[0].id=='cache2'
            ref=ast.parse('scale_active_rows(cache1,local_ids,layer.w13_post_scale)\nsilu_and_mul(cache1,cache2,expert_ids=local_ids.flatten(),expert_step=1)').body
            fn.body[i-1:i+1]=[ref[0],alloc,ref[1]];count+=1;break
    assert count==1 and ast.dump(changed)==ast.dump(original)
    checks.append({'file':rel,'full_file_ast_equals_baseline_after_reversing_only_fusion':True})
for path in DEST.rglob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
for rel in manifest['sources']:
    if not (BASE/rel).exists():assert max(map(len,(DEST/rel).read_text(encoding='utf-8').splitlines()))<=400
report={'status':'EXPERIMENTAL_FULL_SOURCE_SNAPSHOT_LOCAL_GATES_PASS_NO_PERFORMANCE_PROMOTION','baseline_patch_sha256':sha(BASE.parent/'candidate.diff'),
 'sources':{str(p.relative_to(DEST)).replace('\\','/'):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()},'overlay_source_hashes':manifest['sources'],'structural_checks':checks,
 'evidence':{},'scope':'Complete scored baseline source snapshot plus exact scaled-SiLU overlay; all canonical weights/GEMM/partitions/finalizer unchanged. No remote submission or serving score for this fusion.'}
for name,path in [('pair',P/'scale_activation_pair_run/results.json'),('fullchain_partial',P/'scale_activation_fullchain_retry/results.json'),('remaining_numerical',P/'scale_activation_numerical_remaining/results.json'),('offline_sm90',P/'scale_activation_offline_sm90/report.json'),('exact_export',P/'scale_activation_exact_export_retry/results.json'),('metadata_guards',OUT/'guard_audit.json')]:
    report['evidence'][name]={'path':str(path.relative_to(P)).replace('\\','/'),'sha256':sha(path)}
(OUT/'full_snapshot_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
manifest['status']='EXACT_EXPORT_LOCAL_GATES_PASS_EXPERIMENTAL_NOT_PROMOTED_NOT_SUBMITTED'
manifest['scope']='Exact exported wrapper path and guards executed with isolated CUDA filename relocation;16bitwise checks and2fallback checks passed;15native metadata guard cases passed. All4source files frozen and unchanged. Complete clean source snapshot in full_candidate_sources_clean. Earlier incomplete full_candidate_sources contains copied bytecode and is not a packaging input.'
manifest['validation'].extend(['Exact-export16bitwise+2fallback checks passed exit0','Native metadata predicate15cases passed','Entire modified production files AST-identical after reversing only scale+activation fusion'])
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'source_files':len(report['sources']),'snapshot_manifest_sha256':sha(OUT/'full_snapshot_manifest.json'),'status':report['status']}))