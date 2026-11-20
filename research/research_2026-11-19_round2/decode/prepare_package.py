"""Generate isolated package from frozen round-one prologue."""
import hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1]
old=root/'research_2026-11-19/decode/fa3_prologue_kernel.py'
s=old.read_text()
s=s.replace('norm=((ab + 0xc407ffff + odd) >> 20) & 255', '# Explicit modulo avoids debug overflow traps for intended uint32 wrapping.\n    rounded=(ab.to(tl.uint64) + 0xc407ffff + odd.to(tl.uint64)) & 0xffffffff\n    norm=(rounded >> 20) & 255')
s=s.replace('def encode_fp8(x):','def encode_fp8(x, NATIVE: tl.constexpr = False):\n    if NATIVE:\n        return x.to(tl.float8e4nv).to(tl.uint8, bitcast=True)')
s=s.replace('def conversion_probe(X,Y,N:tl.constexpr):','def conversion_probe(X,Y,N:tl.constexpr,NATIVE:tl.constexpr=False):')
s=s.replace('encode_fp8(x),ix<N','encode_fp8(x,NATIVE),ix<N')
s=s.replace('    FP8_MAX: tl.constexpr = 448.0,','    FP8_MAX: tl.constexpr = 448.0,\n    NATIVE: tl.constexpr = False,')
s=s.replace('encode_fp8(x), mask=','encode_fp8(x,NATIVE), mask=')
s=s.replace('encode_fp8(vraw), mask=','encode_fp8(vraw,NATIVE), mask=')
s=s.replace('    kv_mask = mask & (loc != 0)[:, None] & is_k[:, None]',
 '''    valid_loc = (loc >= 0) & (loc < kv_capacity)
    tl.device_assert(valid_loc | ~row_mask | ~is_k, "FA3 prologue cache location out of bounds")
    kv_mask = mask & (loc != 0)[:, None] & is_k[:, None] & valid_loc[:, None]''')
package=here/'prologue_package';package.mkdir(exist_ok=True)
(package/'kernel.py').write_text(s)
(package/'__init__.py').write_text('from .runtime import Fa3Prologue\n')
(here/'source_provenance.json').write_text(json.dumps(dict(round1_source=str(old),round1_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
 kernel_sha256=hashlib.sha256((package/'kernel.py').read_bytes()).hexdigest(),
 differences=['optional separately-qualified native FP8 conversion','explicit uint64 add and modulo2^32 for software conversion under debug checks','native-store-compatible bounds assertion; invalid destinations masked before trapping','same18-head RMS/RoPE arithmetic and explicit BF16 boundaries']),indent=2))
print('prepared',package)
