"""Syntax, patch application, and source-binding validation only."""
import ast,hashlib,json,os,subprocess
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1];folder=here/'deferred_sync_integration'
env=dict(os.environ,GIT_INDEX_FILE=str(folder/'validation.index'));cmd=['git','-C',str(root/'sglang')]
subprocess.run(cmd+['read-tree','297e54fd105e974cdf401e5b4d20253a22c16f90'],env=env,check=True,capture_output=True)
r=subprocess.run(cmd+['apply','--cached','--check',str(folder/'candidate.diff')],env=env,capture_output=True,text=True)
syntax=[]
for p in (folder/'sources').rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'));syntax.append(str(p.relative_to(folder)))
old=ast.parse((here/'dflash_sources/dflash_worker_v2.py').read_text(encoding='utf-8'))
new=ast.parse((folder/'sources/python/sglang/srt/speculative/dflash_worker_v2.py').read_text(encoding='utf-8'))
def methods(tree):
 cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and any(isinstance(f,ast.FunctionDef) and f.name=='_accept_block' for f in x.body))
 return {x.name:ast.dump(x,include_attributes=False) for x in cls.body if isinstance(x,ast.FunctionDef)}
a,b=methods(old),methods(new);changed=[k for k in a if a[k]!=b[k]]
assert set(changed)=={'__init__','_accept_block','forward_batch_generation'},changed
doc=dict(apply_check=r.returncode==0,stderr=r.stderr,syntax=syntax,checkout_mutated=False,changed_worker_methods=changed,
 candidate_sha256=hashlib.sha256((folder/'candidate.diff').read_bytes()).hexdigest(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [here/'deferred_sync.py',here/'deferred_sync_gloo.py',here/'dflash_sources/logits_processor.py']})
(folder/'validation.json').write_text(json.dumps(doc,indent=2));print(json.dumps(doc,indent=2));raise SystemExit(r.returncode)
