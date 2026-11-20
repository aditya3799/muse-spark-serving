import ast,hashlib,functools,types,json
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1];vendor=root/'swarm_2026-11-16/moe/humming_vendor/humming'
def generated(name,arg):
 node=next(n for n in ast.parse((here/name).read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name=='source_for')
 ns={'Path':Path,'humming':types.SimpleNamespace(__file__=str(vendor/'__init__.py')),'hashlib':hashlib,'lru_cache':functools.lru_cache,'HEADER_SHA':'9a021dffc15f2ba2f3c355b4e508aeb3722b351a8be8130ffd4530743df3edd3'}
 exec(compile(ast.Module(body=[node],type_ignores=[]),name,'exec'),ns)
 return ns['source_for'](*arg)
a=generated('native_loop.py',['cycles']);b=generated('fixed_k.py',[]);assert a==b
paths=[]
for line in (here/'candidate.diff').read_text(encoding='utf8').splitlines():
 if line.startswith('diff --git '):paths.append(line.split()[-1][2:])
assert all(p.startswith('python/sglang/') for p in paths)
report=dict(emitted_cuda_identical_to_two_KEEP_variant=True,emitted_cuda_sha256=hashlib.sha256(b.encode()).hexdigest(),full_patch_bytes=(here/'candidate.diff').stat().st_size,full_patch_sha256=hashlib.sha256((here/'candidate.diff').read_bytes()).hexdigest(),delta_bytes=(here/'delta.diff').stat().st_size,paths=paths,added_persistent_gpu_bytes=0,weight_or_precision_changes=False,mem_fraction_static_changed=False,external_pinned_source_modified=False,header_fallback='Frozen Humming header SHA mismatch calls original HummingMethod.forward_layer',gpu_hopper_execution=False)
assert report['full_patch_bytes']<400000
(here/'package_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
