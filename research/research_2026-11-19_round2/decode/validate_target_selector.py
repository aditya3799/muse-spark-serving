import ast,hashlib,json,os,subprocess
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1];folder=here/'target_selector_integration'
env=dict(os.environ,GIT_INDEX_FILE=str(folder/'validation.index'));cmd=['git','-C',str(root/'sglang')]
subprocess.run(cmd+['read-tree','297e54fd105e974cdf401e5b4d20253a22c16f90'],env=env,check=True,capture_output=True)
run=subprocess.run(cmd+['apply','--cached','--check',str(folder/'candidate.diff')],env=env,capture_output=True,text=True)
old=ast.parse((here/'dflash_sources/logits_processor.py').read_text(encoding='utf-8'))
new=ast.parse((folder/'sources/python/sglang/srt/layers/logits_processor.py').read_text(encoding='utf-8'))
def methods(tree):
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LogitsProcessor')
 return {n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
a,b=methods(old),methods(new);changed=[k for k in a if ast.dump(a[k])!=ast.dump(b[k])]
assert changed==['deferred_argmax'],changed
before=a['deferred_argmax'];after=b['deferred_argmax']
assert ast.dump(ast.Module(body=after.body[:-3],type_ignores=[]))==ast.dump(ast.Module(body=before.body[:-2],type_ignores=[]))
assert ast.dump(ast.Module(body=after.body[-2:],type_ignores=[]))==ast.dump(ast.Module(body=before.body[-2:],type_ignores=[]))
helper=folder/'sources/python/sglang/kernels/ops/sampling/dflash_target_select.py';ast.parse(helper.read_text())
root_helper=here.parent/'coordination/target_argmax_postops/target_select.py'
assert helper.read_bytes()==root_helper.read_bytes()
result=dict(apply_check=run.returncode==0,stderr=run.stderr,checkout_mutated=False,changed_methods=changed,
 native_producer_and_collective_ast_unchanged=True,native_fallback_ast_unchanged=True,helper_byte_identical_to_root=True,
 candidate_sha256=hashlib.sha256((folder/'candidate.diff').read_bytes()).hexdigest(),helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest())
(folder/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));raise SystemExit(run.returncode)
