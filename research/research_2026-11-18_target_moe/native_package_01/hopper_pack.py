"""Transform row-major NVFP4 codes and group scales into the Hopper expert tile order."""
import torch
import triton
import triton.language as tl


@triton.jit
def _row(physical, gate_up: tl.constexpr):
    if gate_up:
        return (physical // 16) * 8 + physical % 8 + (physical % 16 // 8) * 4096
    return physical


@triton.jit
def _pack(W, S, OUT, N:tl.constexpr, GATE_UP:tl.constexpr, B:tl.constexpr):
    tile = tl.program_id(1) + tl.program_id(2)*((N//256)*64)
    quad = tile % 4
    step = tile // 4 % 16
    unit = tile // 64 % (N//256)
    expert = tile // ((N//256)*64)
    dest = OUT + tile*9216
    idx = tl.program_id(0)*B + tl.arange(0,B)
    grp = idx//512
    warp = idx//128%4
    g = idx//16%8
    j = idx//4%4
    t = idx%4
    base = unit*256+quad*64+warp*16+g
    k = step*256+grp*64+j*16+t*2
    word = tl.full((B,),0,tl.uint32)
    for r in tl.static_range(4):
        row = _row(base+(r%2)*8,GATE_UP)
        pair = tl.load(W+(expert*N+row)*2048+(k+(r//2)*8)//2).to(tl.uint32)
        for h in tl.static_range(2):
            code = (pair>>(4*h)) & 15
            if r < 3:
                word |= (code & 7) << ((6+3*r+16*h)%32)
                word |= ((code>>3)&1) << ((15+3*r+16*h)%32)
                # The magnitude field for r=3 is handled separately below.
            else:
                word |= (((code&1)<<4) | ((code>>1)&3) | (code&8)) << (16*h)
    tl.store(dest.to(tl.pointer_type(tl.uint32))+idx,word)
    if tl.program_id(0)*B < 1024:
        group = idx//8
        physical = unit*256+quad*64+(group//8%4)*16+(idx%2)*8+group%8
        row = _row(physical,GATE_UP)
        sc = tl.load(S+(expert*N+row)*256+step*16+(group//32)*4+idx//2%4)
        tl.store(dest+8192+idx,sc)


def pack_experts(weight, scales, gate_up):
    experts, n, k2 = weight.shape
    if (n,k2) != ((8192 if gate_up else 4096),2048) or scales.shape != (experts,n,256):
        raise ValueError("unsupported expert dimensions")
    if weight.dtype != torch.uint8 or scales.dtype != torch.float8_e4m3fn:
        raise ValueError("expected NVFP4 weights and E4M3 block scales")
    tiles = experts*(n//256)*64
    out = torch.empty((tiles,9216),device=weight.device,dtype=torch.uint8)
    _pack[(16,(n//256)*64,experts)](weight.contiguous(),scales.contiguous().view(torch.uint8),out,n,gate_up,128)
    return out
