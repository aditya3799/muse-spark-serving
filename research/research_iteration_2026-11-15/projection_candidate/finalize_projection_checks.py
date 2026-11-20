from pathlib import Path
import ast, hashlib, json, os, shutil, subprocess

root=Path('/mnt/d/OpenKernelChallenge/serving')
out=root/'research_iteration_2026-11-15/projection_candidate'
repo=root/'sglang'
env=os.environ.copy()
env['GIT_INDEX_FILE']=str(out/'validation.index')
base='297e54fd105e974cdf401e5b4d20253a22c16f90'
def run(args):
    return subprocess.check_output(['git',*args],cwd=repo,env=env,text=True).strip()
run(['read-tree',base])
run(['apply','--cached','--check',str(out/'candidate.diff')])
run(['apply','--cached',str(out/'candidate.diff')])
run(['diff','--cached','--check',base])
checks={}
for path in (out/'candidate_sources').rglob('*.py'):
    relative=str(path.relative_to(out/'candidate_sources'))
    actual=subprocess.check_output(['git','show',':'+relative],cwd=repo,env=env)
    assert actual==path.read_bytes(),relative
    ast.parse(actual)
    checks[relative]=hashlib.sha256(actual).hexdigest()
for script in ('build_projection_candidate.py','validate_projection_candidate.py','finalize_projection_checks.py'):
    shutil.copy2(Path('/mnt/c/Windows/Temp')/script,out/script)
report=dict(patch_applies_to_pinned_base=True,source_files=checks,
    staged_tree=run(['write-tree']),diffstat=run(['diff','--cached','--stat',base]),
    active_checkout_modified=False,
    memory_live_tensor_bound='At M=16384/N=8192/TP8: gathered rank-major 256 MiB, reordered output 256 MiB, local output 32 MiB; stock output 256 MiB. Extra live tensor storage <=288 MiB for this projection, excluding GEMM/NCCL workspaces and graph pool effects. Full-model peak requires judge validation.',
    limitations=['No local multi-GPU NCCL/CUDA-graph test','No H200 timing','No full-model correctness or memory check'])
(out/'static_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
