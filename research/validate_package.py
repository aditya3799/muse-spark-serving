"""Static packaging checks and memory bounds for the two-file candidate."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT/'sglang'
files=['python/sglang/srt/models/muse_spark.py','python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py']
for file in files: ast.parse((REPO/file).read_text(encoding='utf-8'))
patch=(ROOT/'patch.diff').read_bytes()
assert patch and len(patch)<=400000
text=patch.decode()
paths=[line.split(' b/',1)[1] for line in text.splitlines() if line.startswith('diff --git ')]
assert set(paths)==set(files),paths
assert all(p.startswith('python/sglang/') and '/model_loader/' not in p for p in paths)
assert not any(x in text for x in ['GIT binary patch','rename from ','rename to ','new mode ','old mode '])
added=[line[1:] for line in text.splitlines() if line.startswith('+') and not line.startswith('+++')]
assert max(map(len,added))<=400
assert not any(x in text for x in ['bfloat16'+'\n']) # no dtype declarations introduced by this patch
# Persistent packed router weights replace, rather than duplicate, old hi/lo caches.
k,n=8192,256
old_cache=2*k*n*2; new_cache=k*(2*n)*2
assert old_cache==new_cache
# New output: 2MN FP32 GEMM + MN FP32 add; old: MN FP32 GEMM + MN FP32 GEMM.
# Both retain one MN result. Conservative extra transient storage is one MN FP32.
m=16384
memory={'router_old_cache_bytes_per_layer':old_cache,'router_new_cache_bytes_per_layer':new_cache,'router_cache_delta_all_layers_bytes':0,'extra_router_temporary_bytes_at_16k_tokens':m*n*4,'alignment_metadata':'reduced or unchanged; no expert-weight repacks added','scope':'component allocation accounting, not a measured full-server peak'}
report={'patch_bytes':len(patch),'sha256':hashlib.sha256(patch).hexdigest(),'changed_files':paths,'max_added_line_characters':max(map(len,added)),'added_lines':len(added),'python_ast':'pass','patch_policy':'pass','memory':memory}
(ROOT/'validation/package.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

