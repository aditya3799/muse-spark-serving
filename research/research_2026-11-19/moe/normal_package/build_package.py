from pathlib import Path
import ast,difflib,hashlib,json,re,subprocess,tempfile
here=Path(__file__).resolve().parent
root=here.parents[2]
parent=(root/'release_2026-11-18/candidate.diff').read_bytes()
assert hashlib.sha256(parent).hexdigest()=='8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b'
base='297e54fd105e974cdf401e5b4d20253a22c16f90'
def diff(name,old,new):
 return 'diff --git a/{0} b/{0}\n'.format(name)+('' if old else 'new file mode 100644\n')+''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),'a/'+name if old else '/dev/null','b/'+name))
sections=re.split(r'(?=^diff --git )',parent.decode(),flags=re.MULTILINE)[1:]
originals={};delta=[]
for j,section in enumerate(sections):
 name=section.splitlines()[0].split()[2][2:]
 old=subprocess.run(['git','-C',str(root/'sglang'),'show',base+':'+name],capture_output=True)
 originals[name]=old.stdout.decode() if old.returncode==0 else ''
 if not name.endswith('/muse_spark_nvfp4_moe.py'):continue
 scored=(root/'release_2026-11-18/candidate_sources'/name).read_text()
 needle='    prepare_large_prefill_partition(layer, torch)\n    return layer'
 assert scored.count(needle)==1
 changed=scored.replace(needle,'    prepare_large_prefill_partition(layer, torch)\n    from sglang.kernels.ops.moe.muse_spark_grouped_gate import prepare_build\n    prepare_build(layer)\n    return layer')
 changed=changed.replace('def _humming_expert_outputs(layer,','def _humming_expert_outputs_reference(layer,')
 changed+='\n\ndef _humming_expert_outputs(layer, x, local_ids, torch, retain=False, *,\n                            defer_downscale=False, global_experts=None):\n    from sglang.kernels.ops.moe.muse_spark_grouped_gate import try_grouped\n    result = try_grouped(_humming_expert_outputs_reference, layer, x, local_ids, torch,\n        retain=retain, defer_downscale=defer_downscale, global_experts=global_experts)\n    if result is not None:\n        return result\n    return _humming_expert_outputs_reference(layer, x, local_ids, torch, retain=retain,\n        defer_downscale=defer_downscale, global_experts=global_experts)\n'
 ast.parse(changed)
 path=here/'sources'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(changed.encode())
 delta.append(diff(name,scored,changed));sections[j]=diff(name,originals[name],changed)
for f in sorted(here.glob('muse_spark_grouped_*.py')):
 text=f.read_text().rstrip()+'\n';ast.parse(text)
 name='python/sglang/kernels/ops/moe/'+f.name
 sections.append(diff(name,'',text));delta.append(sections[-1])
 out=here/'sources'/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(text.encode())
bundle=''.join(sections).encode();assert len(bundle)<400000
max_added_line=max(len(line[1:]) for line in bundle.decode().splitlines() if line.startswith('+') and not line.startswith('+++'))
assert max_added_line<=400,max_added_line
with tempfile.TemporaryDirectory(dir=here) as t:
 tree=Path(t);subprocess.run(['git','-C',t,'init','-q'],check=True)
 for name,old in originals.items():
  if old:
   path=tree/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(old.encode())
 subprocess.run(['git','-C',t,'apply','--check','--whitespace=error','-'],input=bundle,check=True)
 subprocess.run(['git','-C',t,'apply','--whitespace=error','-'],input=bundle,check=True)
 for f in (here/'sources').rglob('*.py'):
  assert (tree/f.relative_to(here/'sources')).read_bytes()==f.read_bytes()
(here/'candidate.diff').write_bytes(bundle)
(here/'delta.diff').write_bytes(''.join(delta).encode())
receipt=dict(max_added_line=max_added_line,bytes=len(bundle),sha256=hashlib.sha256(bundle).hexdigest(),baseline_sha256=hashlib.sha256(parent).hexdigest(),paths=[s.splitlines()[0].split()[2][2:] for s in sections],git_apply_check=True,ast=True)
(here/'package.json').write_text(json.dumps(receipt,indent=2));print(receipt)
