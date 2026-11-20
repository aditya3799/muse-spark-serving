"""Separate hardware-cache derivative; preserve the measured persistent v1 export."""
import ast, difflib, hashlib, json
from pathlib import Path
P = Path(__file__).resolve().parent
REL = Path('python/sglang/kernels/ops/attention/metadata.py')
source = P / 'persistent_metadata/candidate_sources' / REL
assert hashlib.sha256(source.read_bytes()).hexdigest() == '2125af76349ba0ceda9f466ab0a1ab6589e2ad23abfd96547b7ddf20ee60618e'
old = source.read_text(encoding='utf-8')
new = old.replace('from typing import Optional\n', 'from functools import lru_cache\nfrom typing import Optional\n', 1)
marker = '\ndef normal_decode_set_metadata(\n'
helper = '''
@lru_cache(maxsize=None)
def _decode_metadata_sm_count(device_index: int) -> int:
    # CUDA tensor device indices are explicit; never consult the current device.
    assert device_index is not None
    return torch.cuda.get_device_properties(device_index).multi_processor_count

'''
assert new.count(marker) == 1
new = new.replace(marker, '\n' + helper + marker, 1)
old_line = '        sm_count = torch.cuda.get_device_properties(seq_lens.device).multi_processor_count\n'
assert new.count(old_line) == 1
new = new.replace(old_line, '        sm_count = _decode_metadata_sm_count(seq_lens.device.index)\n')
new = new.replace('        # CUDA properties are cached by PyTorch; no tensor values or host lengths\n',
                  '        # Hardware count is cached by explicit tensor device; no tensor values or host lengths\n')
ao = {n.name: n for n in ast.parse(old).body if isinstance(n, ast.FunctionDef)}
an = {n.name: n for n in ast.parse(new).body if isinstance(n, ast.FunctionDef)}
assert set(an) - set(ao) == {'_decode_metadata_sm_count'}
for name in ao:
    if name != 'normal_decode_set_metadata': assert ast.dump(ao[name]) == ast.dump(an[name])
out = P / 'persistent_metadata_cached'
target = out / 'candidate_sources' / REL
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes(new.encode('utf-8'))
diff = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='persistent_v1/' + REL.as_posix(), tofile='cached/' + REL.as_posix()))
(out / 'delta_review.diff').write_bytes(diff.encode('utf-8'))
manifest = dict(status='CPU_HARDWARE_CACHE_DERIVATIVE_READY_FOR_REVIEW',
                original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                candidate_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                kernel_ast_identical=True, frozen_v1_preserved=True,
                cache_key='Explicit seq_lens.device.index; None rejected, no current-device lookup',
                precision='Integer metadata only, original scalar prefix and all page mappings unchanged')
(out / 'manifest.json').write_bytes((json.dumps(manifest, indent=2) + '\n').encode('utf-8'))
print(json.dumps(manifest))
