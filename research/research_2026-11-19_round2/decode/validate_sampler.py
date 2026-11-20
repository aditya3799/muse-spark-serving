import ast,hashlib,json,os,subprocess
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1];folder=here/'sampler_integration'
env=dict(os.environ,GIT_INDEX_FILE=str(folder/'validation.index'));cmd=['git','-C',str(root/'sglang')]
subprocess.run(cmd+['read-tree','297e54fd105e974cdf401e5b4d20253a22c16f90'],env=env,check=True,capture_output=True)
r=subprocess.run(cmd+['apply','--cached','--check',str(folder/'candidate.diff')],env=env,capture_output=True,text=True)
syntax=[]
for p in (folder/'sources').rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'));syntax.append(str(p.relative_to(folder)))
doc=dict(apply_check=r.returncode==0,stderr=r.stderr,syntax=syntax,checkout_mutated=False,candidate_sha256=hashlib.sha256((folder/'candidate.diff').read_bytes()).hexdigest())
(folder/'validation.json').write_text(json.dumps(doc,indent=2));print(json.dumps(doc,indent=2));raise SystemExit(r.returncode)
