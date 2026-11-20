"""Prepare reviewed source-only snapshots; never edits scored or active sources."""
import ast,copy,difflib,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
ROOT=P.parents[2]
BASE=ROOT/'coordination/round5_humming_adaptive_fused/candidate_sources'
GLUE=P.parent/'glue_only_ablation/candidate_sources'
BACK='python/sglang/srt/layers/attention/flashattention_backend.py'
HELP='python/sglang/srt/layers/attention/fa3_swa_scheduler.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected=json.loads((ROOT/'coordination/phase2/PLAN.json').read_text(encoding='utf-8'))['baseline_source_hashes']
assert len(expected)==10
for rel,digest in expected.items():assert sha(BASE/rel)==digest,(rel,'baseline pin changed')
assert sha(GLUE/BACK)=='0b0d0d4288e7bab6c7c4fa3ec9c1068c75e52ddd019ff9eebf24cb3790b50b9e'
swa=(P/'candidate_sources'/BACK).read_text(encoding='utf-8')
glue=(GLUE/BACK).read_text(encoding='utf-8')
node=next(n for n in ast.walk(ast.parse(glue)) if isinstance(n,ast.FunctionDef) and n.name=='glue_draft_metadata')
method=''.join(glue.splitlines(True)[node.lineno-1:node.end_lineno])
marker='    def validate_elastic_cuda_graph_recapture(self) -> None:'
assert swa.count(marker)==1
combined=swa.replace(marker,method+'\n\n'+marker,1)
# Removing only the additive glue method restores exact standalone AST.
a=ast.parse(combined)
cls=next(n for n in a.body if isinstance(n,ast.ClassDef) and n.name=='FlashAttentionBackend')
cls.body=[n for n in cls.body if not(isinstance(n,ast.FunctionDef) and n.name=='glue_draft_metadata')]
assert ast.dump(a)==ast.dump(ast.parse(swa))
for name,text in [('standalone_snapshot',swa),('glue_swa_snapshot',combined)]:
    dest=P/name;src=dest/'candidate_sources';rows=[]
    for rel,digest in expected.items():
        target=src/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if rel==BACK:target.write_bytes((P/'candidate_sources'/BACK).read_bytes() if name=='standalone_snapshot' else text.encode('utf-8'))
        else:target.write_bytes((BASE/rel).read_bytes())
        rows.append(dict(path=rel,baseline_sha256=digest,candidate_sha256=sha(target),changed=sha(target)!=digest))
    target=src/HELP
    original_helper=(P/'candidate_sources'/HELP).read_bytes()
    normalized_helper=original_helper.replace(b'\r\n',b'\n')
    assert ast.dump(ast.parse(original_helper))==ast.dump(ast.parse(normalized_helper))
    target.write_bytes(normalized_helper)
    rows.append(dict(path=HELP,baseline_sha256=None,candidate_sha256=sha(target),changed=True))
    assert len(rows)==11 and sum(row['changed'] for row in rows)==2
    manifest=dict(status='EXPERIMENTAL_SOURCE_ONLY_UNPROMOTED_UNSUBMITTED',baseline_job='e399c79bb1994630',baseline_speedup=1.2237,mechanism='FA3 fixed-width native-SWA per-forward scheduler reuse'+(' plus separate FA4 DFlash draft metadata glue eligibility' if name.startswith('glue') else ''),sources=rows,native_results_sha256=sha(P/'native_run/results.json'),native_probe_sha256=sha(P/'native_probe.py'),helper_cpu_tests_sha256=sha(P/'cpu_lifecycle_validation.json'),combination_ast_check='Removing glue_draft_metadata exactly restores standalone AST',limitations=['SM86 BF16 helper/native calls tested, full backend/model graph not executed','H200 no-split persistent semaphore reset cost not measured','FP8 and distributed serving not tested','FA4 glue runtime not tested on supported hardware','No whole-model score or speedup claim'])
    manifest['helper_export_normalization']={'operation':'CRLF to LF only; tested source untouched','tested_original_sha256':sha(P/'candidate_sources'/HELP),'export_sha256':sha(target),'ast_identical':True}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    delta=''.join(difflib.unified_diff((BASE/BACK).read_text(encoding='utf-8').splitlines(True),text.splitlines(True),fromfile='scored/'+BACK,tofile='candidate/'+BACK))
    delta+=''.join(difflib.unified_diff([],target.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile='candidate/'+HELP))
    (dest/'delta_review.diff').write_text(delta,encoding='utf-8')
    print(name,sha(dest/'manifest.json'),sha(src/BACK))
