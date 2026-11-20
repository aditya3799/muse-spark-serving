"""Preserve measured helper bytes and prove startup additions leave tested math intact."""
import ast,hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent
doc=json.loads((here/'sampler_gpu_9817351.json').read_text())
current=(here/'packed_sampler.py').read_bytes()
prefix=b'"""Optional helper; does not change dense logits or their local maximum reduction."""\nimport torch,triton\nimport triton.language as tl\nfrom packed_sampler_kernel import pack_max_id,select_first_rank\n\n'
old=prefix+current[current.index(b'class PackedDraftCollective:'):]
oldsha=hashlib.sha256(old).hexdigest();assert oldsha==doc['source_sha256']['packed_sampler.py'],(oldsha,doc['source_sha256']['packed_sampler.py'])
(here/'sampler_measured_runtime.py').write_bytes(old)
dump=lambda data:ast.dump(next(n for n in ast.parse(data).body if isinstance(n,ast.ClassDef) and n.name=='PackedDraftCollective'),include_attributes=False)
assert dump(old)==dump(current)
record=dict(measured_runtime_sha256=oldsha,current_runtime_sha256=hashlib.sha256(current).hexdigest(),class_ast_unchanged=True,
 kernel_unchanged=hashlib.sha256((here/'packed_sampler_kernel.py').read_bytes()).hexdigest()==doc['source_sha256']['packed_sampler_kernel.py'],
 additions='CPU startup agreement/allocation vote only; actual kernel launch and buffer class unchanged')
(here/'sampler_math_binding.json').write_text(json.dumps(record,indent=2));print(record)
