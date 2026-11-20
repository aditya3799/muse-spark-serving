"""Verify the final stricter-policy rejection and its exact source delta."""
import hashlib,json
from pathlib import Path
import torch
from output_n_package import runtime_projection as runtime
HERE=Path(__file__).resolve().parent;path=HERE/'output_n_package/runtime_projection.py'
block='''    # Native PyTorch restricts Lt reduction schemes under stricter BF16
    # policies. This bridge intentionally supports only the tested native
    # default; do not silently re-enable reduced-precision partial reductions.
    if torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is not True:
        return None,{"reason":"native BF16 reduction policy requires an unsupported Lt preference mask"}
'''
text=path.read_text();assert text.count(block)==1
previous=text.replace(block,'')
if b'\r\n' in path.read_bytes():previous=previous.replace('\n','\r\n')
previous_hash=hashlib.sha256(previous.encode()).hexdigest()
tested=json.loads((HERE/'output_n_package_probe_v2.json').read_text())['package_sources']['runtime_projection.py']
assert previous_hash==tested,(previous_hash,tested)
saved=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
try:
 torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 plan,metadata=runtime.discover([],8)
 assert plan is None and 'preference mask' in metadata['reason']
 torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
 assert runtime.discover([],8)==(None,{'reason':'no calibration examples'})
finally:torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=saved
doc=dict(correct=True,scope=__doc__,tested_gpu_runtime_sha256=tested,
 final_runtime_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
 delta='Only the explicit native-policy early rejection is added. The True-policy executed path and every other package source are unchanged from package_probe_v2.',
 primary_source='https://github.com/pytorch/pytorch/blob/8b97977cc95f31857c67ce324523254fffde5e39/aten/src/ATen/cuda/CUDABlas.cpp#L301',
 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(HERE/'output_n_policy_gate.json').write_text(json.dumps(doc,indent=2));print(json.dumps(doc,indent=2))
