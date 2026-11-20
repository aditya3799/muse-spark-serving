"""Assemble an isolated cumulative candidate from pinned research artifacts."""
import ast, difflib, hashlib, json, shutil, subprocess
from pathlib import Path
from datetime import datetime, timezone
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE='297e54fd105e974cdf401e5b4d20253a22c16f90'
SCORED=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused'
EXPORT=ROOT/'swarm_2026-11-17/deep_research/expert_partition/candidate_export'
SPLIT='python/sglang/kernels/ops/moe/muse_spark_humming_split.py'
LOADER='python/sglang/srt/layers/muse_spark/muse_spark_nvfp4_moe.py'
SHA='8b3ed81e053e4ccb2a043afd25cb94bb49017995ece517bb35aeb6e1b79e026b'
def digest(raw):return hashlib.sha256(raw).hexdigest()
def read_json(p):return json.loads(p.read_text(encoding='utf-8-sig'))
assert digest((SCORED/'candidate.diff').read_bytes())==SHA
reference=read_json(SCORED/'package_validation.json')
contents={}
for rel,sha in reference['source_hashes'].items():
    raw=(SCORED/'candidate_sources'/rel).read_bytes()
    assert digest(raw)==sha,rel
    contents[rel]=raw
original=contents[SPLIT].decode()
export=(EXPORT/SPLIT).read_text(encoding='utf-8-sig')
assert digest((EXPORT/SPLIT).read_bytes())=='8249ee7f46a1e356935b439df22aa6b411919f45efffb045ad888b91806e9afc'
admission=(HERE/'large_partition_admission.py.txt').read_text(encoding='utf-8-sig')
selector=(HERE/'large_partition_selector.py.txt').read_text(encoding='utf-8-sig')
# Replace only the new selector in the preserved-prefix export.
tree=ast.parse(export)
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_choose_large_prefill_configs')
lines=export.splitlines(True)
new=''.join(lines[:node.lineno-1])+selector.rstrip()+'\n\n'+''.join(lines[node.end_lineno:])
new=new.replace('# Research export: disabled until actual target-device qualification.\n    # This opt-in layer attribute is not enabled by any server/source change.', '# Large-route admission is fixed during canonical weight preparation.')
new=new.replace('# Isolated research alternative; the scored implementation above is preserved.', '# Scalable large-route path; the existing small-route implementation is preserved.')
new+='\n\n'+admission.rstrip()+'\n'
assert new.startswith(original)
contents[SPLIT]=new.encode()
loader=contents[LOADER].decode()
anchor="        layer.fp32_configs[stage] = owned\n    return layer"
assert loader.count(anchor)==1
loader=loader.replace(anchor,"        layer.fp32_configs[stage] = owned\n    from sglang.kernels.ops.moe.muse_spark_humming_split import prepare_large_prefill_partition\n    prepare_large_prefill_partition(layer, torch)\n    return layer")
contents[LOADER]=loader.encode()
for rel,raw in contents.items():
    p=HERE/'candidate_sources'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
base_sources={}
chunks=[]
for rel,raw in contents.items():
    result=subprocess.run(['git','show',BASE+':'+rel],cwd=ROOT/'sglang',capture_output=True)
    old=result.stdout if result.returncode==0 else b''
    if result.returncode!=0:assert rel in reference['new_files'],rel
    base_sources[rel]=old
    header='diff --git a/'+rel+' b/'+rel+'\n'
    if not old:header+='new file mode 100644\n'
    diff=''.join(difflib.unified_diff(old.decode().splitlines(True),raw.decode().splitlines(True),fromfile='a/'+rel if old else '/dev/null',tofile='b/'+rel))
    chunks.append(header+diff)
patch=''.join(chunks).encode();(HERE/'candidate.diff').write_bytes(patch)
added=[line[1:] for line in patch.decode().splitlines() if line.startswith('+') and not line.startswith('+++')]
assert len(patch)<=400000 and max(map(len,added))<=400
assert all(p.startswith('python/sglang/') and '/model_loader/' not in p for p in contents)
for marker in ('GIT binary patch','rename from ','rename to '):assert marker not in patch.decode()
check=HERE/('apply_check_'+datetime.now(timezone.utc).strftime('%H%M%S_%f'));check.mkdir()
for rel,raw in base_sources.items():
    if raw:
        p=check/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
commands=[]
for cmd in (['git','apply','--check','--whitespace=error',str(HERE/'candidate.diff')],['git','apply','--whitespace=error',str(HERE/'candidate.diff')]):
    r=subprocess.run(cmd,cwd=check,capture_output=True,text=True);commands.append({'command':cmd,'exit_code':r.returncode,'stderr':r.stderr});assert r.returncode==0,commands[-1]
assert all((check/rel).read_bytes()==raw for rel,raw in contents.items())
for name in ('flags','flags.proven'):(HERE/name).write_bytes(b'--moe-runner-backend humming\n')
fallback=HERE/'proven';fallback.mkdir(exist_ok=True);shutil.copyfile(SCORED/'candidate.diff',fallback/'candidate.diff');(fallback/'flags').write_bytes(b'--moe-runner-backend humming\n')
manifest={'status':'EXPERIMENTAL_INTEGRATED_CANDIDATE_PENDING_TARGET_CHECK','base':BASE,'patch_sha256':digest(patch),'patch_bytes':len(patch),'changed_files':list(contents),'delta_vs_scored':[SPLIT,LOADER],'source_hashes':{p:digest(b) for p,b in contents.items()},'required_flags':{'--moe-runner-backend':'humming'},'small_route_scored_source_prefix_preserved':True,'pinned_base_apply_and_source_match':True,'package_constraints_passed':True,'max_added_line_chars':max(map(len,added)),'apply_commands':commands,'preparation_fragments':{p:digest((HERE/p).read_bytes()) for p in ['large_partition_admission.py.txt','large_partition_selector.py.txt']},'confirmed_parent_score':1.2237,'new_score':None,'actual_H200_tested':False,'full_model_tested':False,'submissions':0,'active_checkout_changed':False}
(HERE/'BUILD_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ('status','patch_sha256','patch_bytes','delta_vs_scored','pinned_base_apply_and_source_match')}))
