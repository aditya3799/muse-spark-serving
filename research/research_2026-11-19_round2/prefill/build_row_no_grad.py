"""Separate unapplied amendment to frozen joint530e row factory."""
import ast,difflib,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
rel=Path('python/sglang/srt/speculative/dflash_projection_row.py')
source=(ROOT/'research_2026-11-19/prefill/integration_joint/sources'/rel).read_text()
anchor='def create_prefill_projection(model,group,capture_rows,*,ordinary_tp,measure=True):'
assert source.count(anchor)==1
changed=source.replace(anchor,'@torch.no_grad()\n'+anchor)
ast.parse(changed)
out=HERE/'row_no_grad_unapplied'
path=out/'sources'/rel
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(changed)
diff=''.join(difflib.unified_diff(source.splitlines(True),changed.splitlines(True),fromfile='a/'+rel.as_posix(),tofile='b/'+rel.as_posix()))
(out/'incremental.diff').write_text(diff)
manifest=dict(scope='Unapplied one-line memory-lifetime amendment relative to frozen joint530e. No numerical algorithm, tensor shape or qualification policy change.',
 source_before_sha256=hashlib.sha256(source.encode()).hexdigest(),source_after_sha256=hashlib.sha256(changed.encode()).hexdigest(),
 patch_sha256=hashlib.sha256(diff.encode()).hexdigest(),bytes=len(diff.encode()),
 proof='../early_projection_unapplied/autograd_lifetime.json',
 requirement='All existing transport/parity/timing qualification executes inside no_grad so deleting generated input actually releases it before full gathered output allocation.')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
