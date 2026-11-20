import ast,hashlib,json,math,random,statistics
from pathlib import Path
here=Path(__file__).resolve().parent
p=here.parent/'coordination/target_argmax_postops/timing.py'
s=p.read_text(encoding='utf-8-sig');tree=ast.parse(s)
ns={'math':math,'random':random,'statistics':statistics}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('gm','summary')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
rows=[dict(native=[100.,100.],candidate=[80.,80.],control=[v,v]) for v in ([50.,200.]*6)]
result=ns['summary'](rows)
r=dict(scope='Synthetic measurement-control counterexample only; no GPU/kernel test',source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),summary=result,duplicate_control_round_ratios=[2.,.5]*6,finding='Mean control ratio alone can hide large alternating drift; component_keep must not imply two-seed confirmation.')
(here/'target_timing_review.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
