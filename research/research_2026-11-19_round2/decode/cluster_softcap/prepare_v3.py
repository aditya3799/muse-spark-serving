import hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent;old=here/'v2';out=here/'v3';out.mkdir(exist_ok=True)
source=(old/'gluon_prototype.py').read_text()
(old/'frontend_failure_v2.json').write_text(json.dumps(dict(complete=False,stage='Gluon broadcast helper',
 error="module triton.experimental.gluon.language has no attribute broadcast_to",native_compiled=True,gpu_execution=False,
 source_sha256=hashlib.sha256(source.encode()).hexdigest()),indent=2))
# cols already has the exact x layout and the original per-tile index plus a
# constant tile*4096. That constant preserves all index comparisons and avoids
# an implicit-layout broadcast helper. Remove the later offset accordingly.
source=source.replace('gl.broadcast_to(local[None,:],(8,4096))','cols').replace('    indices=indices+tiles*4096\n','')
(out/'gluon_prototype.py').write_text(source)
(out/'compile_offline.py').write_bytes((old/'compile_offline.py').read_bytes())
(out/'stable_compile.py').write_bytes((old/'stable_compile.py').read_bytes())
(out/'api_correction.json').write_text(json.dumps(dict(
 installed_api='Gluon3.8 language/__init__.py exports broadcast,expand_dims,reduce,sum,convert_layout; does not export broadcast_to.',
 correction='Use already materialized cols tensor with identical layout; native first-index ordering invariant under tile-local constant4096*tile; no float arithmetic changed.',
 original_failures_retained=True,gpu_execution=False),indent=2))
print('preparedv3 with native tuple reduction and pre-existing explicit index tensor')
