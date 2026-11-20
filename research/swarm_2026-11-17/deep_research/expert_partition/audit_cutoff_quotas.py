import ast,collections,hashlib,json,random
from pathlib import Path
P=Path(__file__).resolve().parent;p=P/'probe_cutoff_coherent.py';tree=ast.parse(p.read_text(encoding='utf-8-sig'))
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');fn=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='make_routes')
# Python3.8 lacks str.removeprefix; execute equivalent literal slice in CPU oracle.
class Compat(ast.NodeTransformer):
 def visit_Call(self,node):
  if isinstance(node.func,ast.Attribute) and node.func.attr=='removeprefix':return ast.copy_location(ast.Subscript(value=node.func.value,slice=ast.Slice(lower=ast.Constant(value=len(node.args[0].value)),upper=None,step=None),ctx=ast.Load()),node)
  return self.generic_visit(node)
fn=Compat().visit(fn);ast.fix_missing_locations(fn);ns={'random':random};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns)
rows=[]
for m in (768,1024):
 for quota in (31,32,33,47,48,49,63,64,65,95,96,97,143,144,145):
  value=ns['make_routes'](m,8,256,8,'local_count_'+str(quota),216+m)
  assert all(len(row)==8 and len(set(row))==8 and min(row)>=0 and max(row)<256 for row in value)
  counts=collections.Counter(v for row in value for v in row if v<32)
  assert all(counts[e]==quota for e in range(32))
  rows.append({'m':m,'quota':quota,'legal_top8_unique':True,'all32localcounts_exact':True})
report={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'CPU_only':True,'rows':rows,'passed':True};(P/'CUTOFF_QUOTA_FIXTURE_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':len(rows),'passed':True}))
