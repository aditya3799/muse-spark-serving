import ast,hashlib,json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;out=P/'candidate_export';manifest=json.loads((out/'manifest.json').read_text());relative=manifest['export_file'];export=out/relative
proto_tree=ast.parse((P/'scalable_partition.py').read_text(encoding='utf-8-sig'));exp_tree=ast.parse(export.read_text())
def get(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
class Normalize(ast.NodeTransformer):
 def visit_FunctionDef(self,node):
  node.name={'_split_large_prefill_expert_outputs':'split_expert_outputs_scalable','_choose_large_prefill_configs':'choose_partition_configs'}.get(node.name,node.name)
  return self.generic_visit(node)
 def visit_Name(self,node):
  node.id={'_choose_large_prefill_configs':'choose_partition_configs'}.get(node.id,node.id);return node
pairs=[('split_expert_outputs_scalable','_split_large_prefill_expert_outputs'),('choose_partition_configs','_choose_large_prefill_configs'),('partition_routes_scalable','partition_routes_scalable')]
rows=[]
for old,new in pairs:
 equal=ast.dump(get(proto_tree,old),include_attributes=False)==ast.dump(Normalize().visit(get(exp_tree,new)),include_attributes=False)
 assert equal,(old,new);rows.append({'prototype':old,'export':new,'AST_identical_after_symbol_rename':equal})
validation=out/'application_check_lf';target=validation/relative
assert not target.exists(),'fresh apply directory required'
target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(manifest['baseline_file']).read_bytes())
commands=[]
for args in (['git','apply','--check',str((out/'incremental.diff').resolve())],['git','apply',str((out/'incremental.diff').resolve())]):
 result=subprocess.run(args,cwd=validation,capture_output=True,text=True);commands.append({'args':args,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr});assert result.returncode==0,commands[-1]
assert target.read_bytes()==export.read_bytes()
report={'source_equivalence':rows,'commands':commands,'applied_export_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'matches_export':True,'no_active_checkout_edit':True}
(out/'application_and_ast_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

