"""Direct bit-only bridges from the resident native WGS expert representation."""
import torch
import triton
import triton.language as tl


@triton.jit
def _physical(n, UP: tl.constexpr):
    if UP:
        return n % 4096 // 8 * 16 + n % 8 + n // 4096 * 8
    return n


@triton.jit
def _code(src, e, n, k, N: tl.constexpr, UP: tl.constexpr):
    p = _physical(n, UP)
    tile = ((e * (N // 256) + p // 256) * 16 + k // 256) * 4 + p % 256 // 64
    ix = k % 256 // 64 * 512 + p % 64 // 16 * 128
    ix += p % 8 * 16 + k % 64 // 16 * 4 + k % 8 // 2
    word = tl.load(src + tile.to(tl.int64) * 2304 + ix).to(tl.uint32)
    r = p // 8 % 2 + (k // 8 % 2) * 2
    h = k % 2
    mag = (word >> ((6 + 3 * r + 16 * h) % 32).to(tl.uint32)) & 7
    sign = ((word >> ((15 + 3 * r + 16 * h) % 32).to(tl.uint32)) & 1) << 3
    special = ((word >> (4 + 16 * h).to(tl.uint32)) & 1)
    special |= ((word >> (16 * h).to(tl.uint32)) & 3) << 1
    special |= (word >> (16 * h).to(tl.uint32)) & 8
    return tl.where(r == 3, special, mag | sign)


@triton.jit
def _scale(src, e, n, g, N: tl.constexpr, UP: tl.constexpr):
    p = _physical(n, UP)
    tile = ((e * (N // 256) + p // 256) * 16 + g // 16) * 4 + p % 256 // 64
    ix = g % 16 // 4 * 256 + p % 64 // 16 * 64
    ix += p % 8 * 8 + g % 4 * 2 + p // 8 % 2
    return tl.load(src + tile.to(tl.int64) * 9216 + 8192 + ix)


@triton.jit
def _normal_codes(src, out, N: tl.constexpr, UP: tl.constexpr, B: tl.constexpr):
    z = tl.program_id(0) * B + tl.arange(0, B)
    e = tl.program_id(1)
    n = z // 2048
    k = z % 2048 * 2
    value = _code(src, e, n, k, N, UP) | (_code(src, e, n, k + 1, N, UP) << 4)
    tl.store(out + e.to(tl.int64) * N * 2048 + z, value)


@triton.jit
def _humming_codes(src, out, N: tl.constexpr, UP: tl.constexpr,
                   WG: tl.constexpr, B: tl.constexpr):
    z = tl.program_id(0) * B + tl.arange(0, B)
    e = tl.program_id(1)
    kb = z // (N * 2)
    nb = z % (N * 2) // 128
    lane = z % 128 // 4
    a = z % 4
    word = tl.full((B,), 0, tl.uint32)
    for i in tl.static_range(2):
        for j in tl.static_range(2):
            pair = j * 2 + i if WG else i * 2 + j
            n = nb * 64 + a * 16 + i * 8 + lane // 4
            k = kb * 16 + j * 8 + lane % 4 * 2
            word |= _code(src, e, n, k, N, UP) << (pair * 4)
            word |= _code(src, e, n, k + 1, N, UP) << (pair * 4 + 16)
    tl.store(out + e.to(tl.int64) * N * 512 + z, word)


@triton.jit
def _scales(src, out, N: tl.constexpr, UP: tl.constexpr,
            NORMAL: tl.constexpr, B: tl.constexpr):
    z = tl.program_id(0) * B + tl.arange(0, B)
    e = tl.program_id(1)
    if NORMAL:
        n = z // 4 % N
        g = z // (4 * N) * 4 + z % 4
    else:
        physical_n = z % N
        n = physical_n // 64 * 64 + physical_n % 8 * 8 + physical_n % 64 // 8
        g = z // N
    value = _scale(src, e, n, g, N, UP)
    tl.store(out + e.to(tl.int64) * N * 256 + z, value)


def convert(wgs, scratch, n, *, gate_up, layout):
    """Write one complete stage in caller-owned scratch; never retain its contents.

    layout is normal, humming_mma or humming_wgmma. No dtype conversion occurs.
    Caller owns serialization and graph-safe workspace lifetimes.
    """
    if n not in (4096, 8192) or bool(gate_up) != (n == 8192):
        raise ValueError('fixed Muse W13/W2 geometry required')
    if layout not in ('normal', 'humming_mma', 'humming_wgmma'):
        raise ValueError('unsupported destination layout')
    per_expert = n * 4096 * 9 // 16
    if (wgs.dtype != torch.uint8 or scratch.dtype != torch.uint8 or
        not wgs.is_contiguous() or not scratch.is_contiguous() or
        not wgs.is_cuda or wgs.device != scratch.device or
        wgs.numel() % per_expert or not 1 <= wgs.numel() // per_expert <= 32):
        raise ValueError('invalid WGS source or scratch storage')
    experts = wgs.numel() // per_expert
    total = experts * per_expert
    if scratch.numel() < total:
        raise ValueError('insufficient shared workspace')
    a0, a1 = wgs.data_ptr(), wgs.data_ptr() + wgs.numel()
    b0, b1 = scratch.data_ptr(), scratch.data_ptr() + total
    if a0 < b1 and b0 < a1:
        raise ValueError('source and destination must not overlap')
    flat = scratch.view(-1)
    codes_bytes = experts * n * 2048
    codes = flat[:codes_bytes]
    scales = flat[codes_bytes:total]
    if layout == 'normal':
        _normal_codes[(n * 2048 // 256, experts)](wgs.view(torch.uint32), codes, n, gate_up, 256)
        _scales[(n * 256 // 256, experts)](wgs, scales, n, gate_up, True, 256)
        return codes.view(experts, n, 2048), scales.view(torch.int32).view(experts, 64, n)
    _humming_codes[(n * 512 // 256, experts)](
        wgs.view(torch.uint32), codes.view(torch.uint32), n, gate_up,
        layout == 'humming_wgmma', 256)
    _scales[(n * 256 // 256, experts)](wgs, scales, n, gate_up, False, 256)
    return codes.view(torch.int32).view(experts, 256, n * 2), scales.view(experts, 256, n)
