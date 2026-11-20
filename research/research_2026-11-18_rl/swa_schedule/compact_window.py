"""Prepare distinct full and bounded sliding-window page tables for FA3 verification."""
import triton
import triton.language as tl


@triton.jit
def _prepare(lengths, cumulative, full, source, request, seq, sliding, mapping, short, short_cu,
             IN_STRIDE: tl.constexpr, FULL_STRIDE: tl.constexpr, SHORT_STRIDE: tl.constexpr,
             LIMIT: tl.constexpr, B: tl.constexpr, ROUND_B: tl.constexpr, Q: tl.constexpr,
             WINDOW: tl.constexpr, ALIGN: tl.constexpr, FULL_BLOCK: tl.constexpr,
             SWA_BLOCK: tl.constexpr, FULL_TILES: tl.constexpr):
    tile = tl.program_id(0)
    row = tl.program_id(1)
    if tile == 0 and row == 0:
        idx = tl.arange(0, ROUND_B)
        active = idx < B
        absolute = tl.load(seq + idx, mask=active, other=0).to(tl.int32) + Q
        absolute = tl.where(active, absolute, 0)
        trimmed = absolute - tl.maximum(0, absolute - WINDOW) // ALIGN * ALIGN
        tl.store(lengths + idx, absolute, mask=active)
        tl.store(short + idx, trimmed, mask=active)
        tl.store(cumulative + idx, tl.cumsum(absolute, 0) - absolute, mask=active)
        tl.store(short_cu + idx, tl.cumsum(trimmed, 0) - trimmed, mask=active)
        tl.store(cumulative+B, tl.sum(absolute, 0))
        tl.store(short_cu+B, tl.sum(trimmed, 0))
    absolute = tl.minimum(LIMIT, tl.load(seq+row).to(tl.int32)+Q)
    req = tl.load(request+row).to(tl.int64)
    if tile < FULL_TILES:
        if tile*FULL_BLOCK < absolute:
            col = tile*FULL_BLOCK + tl.arange(0, FULL_BLOCK)
            page = tl.load(source + req*IN_STRIDE + col, mask=col<absolute, other=0)
            tl.store(full + row*FULL_STRIDE + col, page, mask=col<absolute)
    else:
        base = tl.maximum(0, absolute-WINDOW) // ALIGN * ALIGN
        first = (tile-FULL_TILES)*SWA_BLOCK
        if first < absolute-base:
            col = first + tl.arange(0, SWA_BLOCK)
            live = col < absolute-base
            page = tl.load(source + req*IN_STRIDE + base+col, mask=live, other=0)
            physical = tl.load(mapping + page.to(tl.int64), mask=live, other=0)
            tl.store(sliding + row*SHORT_STRIDE + col, physical, mask=live)


def prepare(lengths,cumulative,full,source,request,limit,seq,delta,page_size,sliding,mapping,
            *,short,short_cu,swa_suffix_tokens,alignment):
    b=seq.numel()
    assert b and page_size==1 and delta>0 and alignment>0
    width=min(limit,swa_suffix_tokens+alignment-1)
    assert sliding.shape==(b,width) and full.shape[0]==b and full.shape[1]>=limit
    assert all(t.is_contiguous() for t in (seq,request,mapping,lengths,cumulative,short,short_cu))
    assert all(t.stride(1)==1 for t in (source,full,sliding))
    ft=triton.cdiv(limit,2048)
    _prepare[(ft+triton.cdiv(width,256),b)](
        lengths,cumulative,full,source,request,seq,sliding,mapping,short,short_cu,
        source.stride(0),full.stride(0),sliding.stride(0),limit,b,triton.next_power_of_2(b),
        delta,swa_suffix_tokens,alignment,2048,256,ft,num_warps=4)
