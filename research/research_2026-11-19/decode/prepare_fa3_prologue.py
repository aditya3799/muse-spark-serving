import hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent
source=here.parents[1]/'sglang/python/sglang/kernels/ops/attention/muse_spark_qk_norm_rope_store.py'
s=source.read_text();s=s[s.index('@triton.jit'):s.index('\ndef _triton_config')]
s=s.replace('_qk_rms_norm_fwd_kernel','fusion18_kernel',1)
s=s.replace('FP8_E4M3_MAX','448.0')
start=s.index('    if HAS_KV_STORE:\n        HEADS:')
end=s.index('    token = rows // HEADS',start)
s=s[:start]+'    HEADS: tl.constexpr = NUM_Q_HEADS + NUM_KV_HEADS\n'+s[end:]
start=s.index('    if HAS_KV_STORE:\n        x = tl.where')
end=s.index('    q_dst =',start)
s=s[:start]+'''    loc = tl.load(CACHE_LOC + token, mask=row_mask, other=0).to(tl.int64)
    kv_mask = mask & (loc != 0)[:, None] & is_k[:, None]
    kv_dst = loc * (NUM_KV_HEADS * D) + (head - NUM_Q_HEADS) * D
    tl.store(K_CACHE + kv_dst[:, None] + cols[None, :], encode_fp8(x), mask=kv_mask)
    vraw = tl.load(QKV + (token * stride_qkv + (head + NUM_KV_HEADS) * D)[:, None] + cols[None, :],
                  mask=mask & is_k[:, None], other=0.0).to(tl.float32)
    tl.store(V_CACHE + kv_dst[:, None] + cols[None, :], encode_fp8(vraw), mask=kv_mask)

'''+s[end:]
s=s.replace('tl.store(Q_OUT + q_dst[:, None] + cols[None, :], x, mask=mask & is_q[:, None])',
 'tl.store(Q_OUT + q_dst[:, None] + cols[None, :], encode_fp8(x), mask=mask & is_q[:, None])')
s=s.replace('    tl.store(K_OUT + k_dst[:, None] + cols[None, :], x, mask=mask & is_k[:, None])','')
prefix='''import triton
import triton.language as tl

@triton.jit
def encode_fp8(x):
    # Installed Torch2.14.1 header-only conversion, saturating RN-even.
    # Software bits permit an honest SM86 test without native FP8 instructions.
    bits=x.to(tl.float32).to(tl.uint32,bitcast=True)
    sign=(bits >> 24) & 128
    ab=bits & 0x7fffffff
    odd=(ab >> 20) & 1
    norm=((ab + 0xc407ffff + odd) >> 20) & 255
    norm=tl.minimum(norm,126)
    tiny=ab.to(tl.float32,bitcast=True)+16384.0
    den=tiny.to(tl.uint32,bitcast=True)-(141 << 23)
    finite=tl.where(ab < (121 << 23),den,norm)
    result=tl.where(ab >= (1087 << 20),tl.where(ab > 0x7f800000,127,126),finite)
    return (result | sign).to(tl.uint8)

@triton.jit
def conversion_probe(X,Y,N:tl.constexpr):
    ix=tl.program_id(0)*128+tl.arange(0,128)
    x=tl.load(X+ix,ix<N,other=0).to(tl.float32)
    tl.store(Y+ix,encode_fp8(x),ix<N)

'''
p=here/'fa3_prologue_kernel.py';p.write_text(prefix+s)
(here/'fa3_prologue_source.json').write_text(json.dumps(dict(native_source=str(source),native_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 candidate_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='Source-derived18head normalization with piggybackVstore and exact softwareFP8cast; not serving integrated.'),indent=2))
print('generated',p.name)
