import ast,hashlib,json
from pathlib import Path
R=Path('/mnt/d/OpenKernelChallenge/serving');H=R/'research_2026-11-19_round2/prefill'
paths=[R/'sglang/python/sglang/kernels/ops/attention/muse_spark_qk_norm_rope_store.py',R/'research_2026-11-18_rl_prefill/decode_qkv/parent_muse_spark_qk_norm_rope_store.py']
records=[]
for p in paths:
 s=p.read_text();tree=ast.parse(s);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_qk_rms_norm_fwd_kernel')
 body=ast.dump(fn,include_attributes=False);records.append(dict(path=str(p),whole_file_sha=hashlib.sha256(p.read_bytes()).hexdigest(),kernel_ast_sha=hashlib.sha256(body.encode()).hexdigest()))
print(json.dumps(records,indent=2));assert records[0]['kernel_ast_sha']==records[1]['kernel_ast_sha']
(H/'prologue_rounding_debug/native_cache_provenance.json').write_text(json.dumps(dict(passed=True,records=records,explanation='M8 cache-hit IR retains older file-location metadata. The actual extracted original kernel AST is identical to the old cached source; large M compiled from current path. Whole files may differ outside that kernel.'),indent=2))
