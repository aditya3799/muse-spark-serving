"""Isolated tile-aligned bridges; only byte permutation, same guarded API."""
import torch
import triton
import triton.language as tl
from wgs_bridge import _physical, _code, _scales

@triton.jit
def _word(src,e,n,k,N:tl.constexpr,UP:tl.constexpr):
    p=_physical(n,UP)
    tile=((e*(N//256)+p//256)*16+k//256)*4+p%256//64
    ix=k%256//64*512+p%64//16*128+p%8*16+k%64//16*4+k%8//2
    return tl.load(src+tile.to(tl.int64)*2304+ix).to(tl.uint32)

@triton.jit
def _decode(word,r,h:tl.constexpr):
    mag=(word>>((6+3*r+16*h)%32).to(tl.uint32))&7
    sign=((word>>((15+3*r+16*h)%32).to(tl.uint32))&1)<<3
    special=((word>>(4+16*h))&1)|(((word>>(16*h))&3)<<1)|((word>>(16*h))&8)
    return tl.where(r==3,special,mag|sign)

@triton.jit
def _normal_tile(src,out,N:tl.constexpr,UP:tl.constexpr,R:tl.constexpr):
    z=tl.program_id(0)
    e=tl.program_id(1)
    n=z//16*R+tl.arange(0,R)[:,None]
    k=z%16*256+tl.arange(0,128)[None,:]*2
    word=_word(src,e,n,k,N,UP)
    p=_physical(n,UP)
    r=p//8%2+(k//8%2)*2
    value=_decode(word,r,0)|(_decode(word,r,1)<<4)
    tl.store(out+e.to(tl.int64)*N*2048+n*2048+k//2,value)

@triton.jit
def _humming_word(src,out,N:tl.constexpr,UP:tl.constexpr,WG:tl.constexpr,B:tl.constexpr):
    z=tl.program_id(0)*B+tl.arange(0,B)
    e=tl.program_id(1)
    kb=z//(N*2);nb=z%(N*2)//128;lane=z%128//4;a=z%4
    n=nb*64+a*16+lane//4;k=kb*16+lane%4*2
    q0=_word(src,e,n,k,N,UP)
    if UP:q1=_word(src,e,n+8,k,N,UP)
    else:q1=q0
    word=tl.full((B,),0,tl.uint32)
    for i in tl.static_range(2):
        q=q0 if i==0 else q1
        parity=n//4096 if UP else i
        for j in tl.static_range(2):
            r=tl.full((B,),parity+2*j,tl.int32) if not UP else parity+2*j
            pair=j*2+i if WG else i*2+j
            word|=_decode(q,r,0)<<(pair*4)
            word|=_decode(q,r,1)<<(pair*4+16)
    tl.store(out+e.to(tl.int64)*N*512+z,word)


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
        _normal_tile[(n // 16 * 16, experts)](wgs.view(torch.uint32), codes, n, gate_up, 16, num_warps=4)
        _scales[(n * 256 // 256, experts)](wgs, scales, n, gate_up, True, 256)
        return codes.view(experts, n, 2048), scales.view(torch.int32).view(experts, 64, n)
    _humming_word[(n * 512 // 256, experts)](
        wgs.view(torch.uint32), codes.view(torch.uint32), n, gate_up,
        layout == 'humming_wgmma', 256)
    _scales[(n * 256 // 256, experts)](wgs, scales, n, gate_up, False, 256)
    return codes.view(torch.int32).view(experts, 256, n * 2), scales.view(experts, 256, n)
