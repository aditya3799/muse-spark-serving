"""Device-only packing of the scored Humming BF16/FP4 operands into Hopper grouped-GEMM arrays."""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack_codes(src, dst, chosen, N: tl.constexpr, K: tl.constexpr, WG: tl.constexpr,
                BN: tl.constexpr, PAIRS: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = (tl.program_id(1) * PAIRS + tl.arange(0, PAIRS)) * 2
    slot = tl.program_id(2)
    expert = tl.load(chosen + slot).to(tl.int64)
    nn, kk = n[:, None], k[None, :]
    lane = (nn % 8) * 4 + (kk % 8) // 2
    word = (expert * (K // 16) + kk // 16) * (N * 2)
    word += nn // 64 * 128 + lane * 4 + (nn % 64) // 16
    if WG:
        pair = ((kk // 8) % 2) * 2 + (nn // 8) % 2
    else:
        pair = ((nn // 8) % 2) * 2 + (kk // 8) % 2
    bits = tl.load(src + word).to(tl.uint32) >> (pair * 4).to(tl.uint32)
    pair_byte = (bits & 15) | ((bits >> 12) & 240)
    tl.store(dst + (slot.to(tl.int64) * N + nn) * (K // 2) + kk // 2, pair_byte)


@triton.jit
def _pack_scales(src, dst, chosen, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    col = tl.program_id(1)
    slot = tl.program_id(2)
    expert = tl.load(chosen + slot).to(tl.int64)
    swizzle = n // 64 * 64 + (n % 8) * 8 + (n % 64) // 8
    base = (expert * (K // 16) + col * 4) * N + swizzle
    word = tl.full((BN,), 0, tl.uint32)
    for i in tl.static_range(4):
        word |= tl.load(src + base + i * N).to(tl.uint32) << (8 * i)
    tl.store(dst + (slot.to(tl.int64) * (K // 64) + col) * N + n, word)


def convert(weights, scales, chosen, scratch, n, k, wgmma):
    groups = chosen.numel()
    codes_bytes = groups * n * k // 2
    total = codes_bytes + groups * n * k // 16
    if n % 64 or k % 64 or scratch.dtype != torch.uint8 or scratch.numel() < total:
        raise ValueError("requires aligned N,K and sufficient byte workspace")
    if not all(a.is_contiguous() for a in (weights, scales, chosen, scratch)):
        raise ValueError("requires contiguous tensors")
    if chosen.dtype != torch.int32 or any(a.device != scratch.device for a in (weights, scales, chosen)):
        raise ValueError("invalid expert list or device")
    if weights.numel()*weights.element_size()%(n*k//2) or scales.numel()*scales.element_size()!=(weights.numel()*weights.element_size())//8:
        raise ValueError("incompatible weight or group-16 scale storage")
    flat = scratch.view(-1)
    codes = flat[:codes_bytes].reshape(groups, n, k//2)
    words = flat[codes_bytes:total].view(torch.int32).reshape(groups, k//64, n)
    _pack_codes[(n//16,k//64,groups)](weights.view(torch.int32), codes, chosen, n, k, wgmma, 16, 32)
    _pack_scales[(n//64,k//64,groups)](scales.view(torch.uint8), words, chosen, n, k, 64)
    return codes, words


@triton.jit
def _rank(ids, excess, dest, S: tl.constexpr, E: tl.constexpr, B: tl.constexpr):
    e, chunk = tl.program_id(0), tl.program_id(1)
    slot = chunk*B+tl.arange(0,B)
    match = tl.load(ids+slot, slot<S, other=-1)==e
    n = tl.cumsum(match.to(tl.int32),0)
    tl.store(dest+slot, n-1, match)
    tl.store(excess+chunk*E+e, tl.sum(match.to(tl.int32),0))


@triton.jit
def _bases(prefix, offset, count, padded, CHUNKS: tl.constexpr, E: tl.constexpr, EB: tl.constexpr):
    e=tl.arange(0,EB)
    n=tl.load(prefix+(CHUNKS-1)*E+e,e<E,other=0)
    length=tl.where(e<E,tl.maximum(16,(n+15)//16*16),0)
    start=tl.cumsum(length,0)-length
    tl.store(offset+e,start,e<E)
    tl.store(count+e,n,e<E)
    tl.store(padded+e,length,e<E)


@triton.jit
def _map(ids, dest, prefix, offset, S:tl.constexpr, E:tl.constexpr, B:tl.constexpr):
    ch=tl.program_id(0)
    i=ch*B+tl.arange(0,B)
    e=tl.load(ids+i,i<S,other=-1)
    live=(e>=0)&(e<E)
    local=tl.load(dest+i,live,other=0)
    prev=tl.load(prefix+(ch-1)*E+e,live&(ch>0),other=0)
    first=tl.load(offset+e,live,other=0)
    tl.store(dest+i,tl.where(live,first+prev+local,-1),i<S)


def route_buffers(ids, experts):
    s=ids.numel()
    chunks=triton.cdiv(s,256)
    kw=dict(device=ids.device,dtype=torch.int32)
    return (torch.empty((chunks,experts),**kw),torch.empty((chunks,experts),**kw),
            torch.empty(s,**kw),*(torch.empty(experts,**kw) for _ in range(3)))


def route(ids, bufs):
    partial,prefix,dest,base,counts,padded=bufs
    s=ids.numel()
    chunks,e=partial.shape
    if not ids.is_contiguous() or ids.dtype!=torch.int32 or ids.ndim!=2 or ids.size(1)!=8 or not s:
        raise ValueError("requires local contiguous int32 top-8 route IDs")
    if chunks!=triton.cdiv(s,256) or dest.numel()!=s:
        raise ValueError("workspace mismatches route shape")
    _rank[(e,chunks)](ids,partial,dest,s,e,256)
    torch.cumsum(partial,dim=0,dtype=torch.int32,out=prefix)
    _bases[(1,)](prefix,base,counts,padded,chunks,e,triton.next_power_of_2(e))
    _map[(chunks,)](ids,dest,prefix,base,s,e,256)
    return dest,base,counts,padded


@triton.jit
def _rows(src,dst,dest,H:tl.constexpr,PACK:tl.constexpr,RESTORE:tl.constexpr,B:tl.constexpr):
    s=tl.program_id(0)
    cols=tl.program_id(1)*B+tl.arange(0,B)
    row=tl.load(dest+s)
    if PACK:
        if row>=0:
            tl.store(dst+row.to(tl.int64)*H+cols,tl.load(src+(s//8).to(tl.int64)*H+cols))
    else:
        val=tl.load(src+row.to(tl.int64)*H+cols,row>=0,other=0.)
        if RESTORE:
            val=(val.to(tl.float32)*4.).to(tl.bfloat16)
        tl.store(dst+s.to(tl.int64)*H+cols,val)


@triton.jit
def _clear_pad(dst,base,counts,padded,H:tl.constexpr,B:tl.constexpr):
    e=tl.program_id(0)
    r=tl.program_id(1)
    col=tl.program_id(2)*B+tl.arange(0,B)
    count=tl.load(counts+e)
    if count+r<tl.load(padded+e):
        row=tl.load(base+e)+count+r
        tl.store(dst+row.to(tl.int64)*H+col,0.)


def gather(x, out, mapping):
    dest,base,counts,padded=mapping
    h=x.size(1)
    b=min(256,h)
    if h%b or x.shape[0]*8!=dest.numel() or out.shape[0]<dest.numel()+16*len(base) or out.shape[1]!=h:
        raise ValueError("invalid source or row workspace")
    _rows[(dest.numel(),h//b)](x,out,dest,h,True,False,b)
    _clear_pad[(len(base),16,h//b)](out,base,counts,padded,h,b)


def scatter(grouped, canonical, dest, restore=True):
    h=grouped.size(1)
    b=min(256,h)
    if h%b or canonical.shape!=(dest.numel(),h):
        raise ValueError("invalid canonical output")
    _rows[(dest.numel(),h//b)](grouped,canonical,dest,h,False,restore,b)
