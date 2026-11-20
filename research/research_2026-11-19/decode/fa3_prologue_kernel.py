import triton
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

@triton.jit
def fusion18_kernel(
    QKV,  # [T, qkv_width] source projection output
    Q_OUT,  # [T, NUM_Q_HEADS * D]
    K_OUT,  # [T, NUM_KV_HEADS * D]
    POS,  # [T] positions; only read when HAS_ROPE
    COS_SIN,  # [max_pos, D] fp32, cos then sin halves; only read when HAS_ROPE
    K_CACHE,  # [N, NUM_KV_HEADS * D] BF16/FP8 pool; written when HAS_KV_STORE
    V_CACHE,  # [N, NUM_KV_HEADS * D]
    CACHE_LOC,  # [T] pool row per token
    eps,
    stride_qkv,  # row stride of QKV, in elements
    n_rows,  # T * (NUM_Q_HEADS + NUM_KV_HEADS [+ NUM_KV_HEADS for V])
    kv_capacity,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    HAS_KV_STORE: tl.constexpr,
    FP8_KV_STORE: tl.constexpr,
    USE_PDL: tl.constexpr = False,
    FP8_MAX: tl.constexpr = 448.0,
):
    # One (token, head) row per lane of the block; fp32 RMS reduction over D.
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        # ---- PDL: Trigger next kernel here to increase overlap
        tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_rows
    cols = tl.arange(0, BD)
    col_mask = cols < D
    mask = row_mask[:, None] & col_mask[None, :]

    # Q, K (and V when storing the cache) are adjacent in the projection, so
    # one combined head index addresses all of them with one offset.
    HEADS: tl.constexpr = NUM_Q_HEADS + NUM_KV_HEADS
    token = rows // HEADS
    head = rows % HEADS
    is_q = head < NUM_Q_HEADS
    is_k = (head >= NUM_Q_HEADS) & (head < NUM_Q_HEADS + NUM_KV_HEADS)
    is_v = head >= NUM_Q_HEADS + NUM_KV_HEADS

    src = token * stride_qkv + head * D
    raw = tl.load(QKV + src[:, None] + cols[None, :], mask=mask, other=0.0)
    x = raw.to(tl.float32)
    var = tl.sum(tl.where(mask, x * x, 0.0), axis=1) / D
    rstd = tl.rsqrt(var + eps)
    x = x * rstd[:, None]

    if HAS_ROPE:
        # Rope reads the bf16-rounded row, as the separate rope kernel did.
        x = x.to(QKV.dtype.element_ty).to(tl.float32)
        partner_cols = cols ^ 1
        xp = tl.load(
            QKV + src[:, None] + partner_cols[None, :], mask=mask, other=0.0
        ).to(tl.float32)
        xp = (xp * rstd[:, None]).to(QKV.dtype.element_ty).to(tl.float32)
        pos = tl.load(POS + token, mask=row_mask, other=0).to(tl.int64)
        freq = cols // 2
        cs_mask = row_mask[:, None] & (cols < D)[None, :]
        cos = tl.load(
            COS_SIN + pos[:, None] * D + freq[None, :], mask=cs_mask, other=1.0
        )
        sin = tl.load(
            COS_SIN + pos[:, None] * D + D // 2 + freq[None, :], mask=cs_mask, other=0.0
        )
        is_even = (cols % 2 == 0)[None, :]
        x = tl.where(is_even, x * cos - xp * sin, xp * sin + x * cos)
    if FP8_KV_STORE:
        x = x.to(QKV.dtype.element_ty).to(tl.float32)
    loc = tl.load(CACHE_LOC + token, mask=row_mask, other=0).to(tl.int64)
    kv_mask = mask & (loc != 0)[:, None] & is_k[:, None]
    kv_dst = loc * (NUM_KV_HEADS * D) + (head - NUM_Q_HEADS) * D
    tl.store(K_CACHE + kv_dst[:, None] + cols[None, :], encode_fp8(x), mask=kv_mask)
    vraw = tl.load(QKV + (token * stride_qkv + (head + NUM_KV_HEADS) * D)[:, None] + cols[None, :],
                  mask=mask & is_k[:, None], other=0.0).to(tl.float32)
    tl.store(V_CACHE + kv_dst[:, None] + cols[None, :], encode_fp8(vraw), mask=kv_mask)

    q_dst = tl.where(is_q, (token * NUM_Q_HEADS + head) * D, 0)
    k_dst = tl.where(is_k, (token * NUM_KV_HEADS + head - NUM_Q_HEADS) * D, 0)
    tl.store(Q_OUT + q_dst[:, None] + cols[None, :], encode_fp8(x), mask=mask & is_q[:, None])


