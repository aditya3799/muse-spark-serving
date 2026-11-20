import ast, hashlib, json
from pathlib import Path

root=Path(__file__).parent
old_path=root/'candidate_before_utf8_fix.py'
new_path=root/'candidate/python/sglang/srt/layers/attention/flashattention_backend.py'
old=ast.parse(old_path.read_text(encoding='utf-8'))
new=ast.parse(new_path.read_text(encoding='utf-8'))
def strip_docs(node):
    for child in ast.iter_child_nodes(node):strip_docs(child)
    if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body.pop(0)
strip_docs(old);strip_docs(new)
strict_equal=ast.dump(old)==ast.dump(new)
old_strings=[n for n in ast.walk(old) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
new_strings=[n for n in ast.walk(new) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
changes=[]
for a,b in zip(old_strings,new_strings):
    if a.value!=b.value:
        restored=a.value.encode('cp1252').decode('utf-8')
        assert restored==b.value
        changes.append(dict(line=a.lineno,old=a.value,new=b.value))
        a.value=restored
assert ast.dump(old)==ast.dump(new)
methods={'_compute_scheduler_metadata','_compute_verify_scheduler_metadata','_apply_cuda_graph_metadata','_init_full_cg_decode_metadata','forward_extend'}
result=dict(passed=True,old_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),new_sha256=hashlib.sha256(new_path.read_bytes()).hexdigest(),runtime_ast_equal_excluding_docstrings=strict_equal,non_docstring_literal_restorations=changes,all_runtime_ast_equal_after_lossless_encoding_restoration=True)
(root/'encoding_validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=True))
print(json.dumps(result,indent=2,ensure_ascii=True))
