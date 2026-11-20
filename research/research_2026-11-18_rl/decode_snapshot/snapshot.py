"""Materialize private full and sliding page tables before the early scheduler event."""
import triton
import triton.language as tl

@triton.jit
def _snapshot(lengths, cumulative, full, source, request, seq, sliding, mapping,
              STRIDE_SOURCE:tl.constexpr, STRIDE_FULL:tl.constexpr, STRIDE_SWA:tl.constexpr,
              LIMIT:tl.constexpr, B:tl.constexpr, ROUND_B:tl.constexpr, DELTA:tl.constexpr,
              WINDOW:tl.constexpr, FULL_BLOCK:tl.constexpr, SWA_BLOCK:tl.constexpr, FULL_TILES:tl.constexpr):
    tile=tl.program_id(0)
    row=tl.program_id(1)
    if tile==0 and row==0:
        idx=tl.arange(0,ROUND_B)
        valid=idx<B
        vals=tl.load(seq+idx,mask=valid,other=0).to(tl.int32)
        vals=tl.where(valid,vals+DELTA,0)
        ends=tl.cumsum(vals,0)
        tl.store(lengths+idx,vals,mask=valid)
        tl.store(cumulative+idx,ends-vals,mask=valid)
        tl.store(cumulative+B,tl.sum(vals,0))
    length=tl.minimum(LIMIT,tl.load(seq+row).to(tl.int32)+DELTA)
    req=tl.load(request+row).to(tl.int64)
    if tile<FULL_TILES:
        first=tile*FULL_BLOCK
        if first<length:
            cols=first+tl.arange(0,FULL_BLOCK)
            slot=tl.load(source+req*STRIDE_SOURCE+cols,mask=cols<length,other=0)
            tl.store(full+row*STRIDE_FULL+cols,slot,mask=cols<length)
    else:
        # Align the first (possibly partially reachable) suffix tile. All live
        # output columns stay at their absolute logical positions for native FA3.
        start=tl.maximum(0,tl.load(seq+row)+DELTA-WINDOW)
        first=(start//SWA_BLOCK+tile-FULL_TILES)*SWA_BLOCK
        if first<length:
            cols=first+tl.arange(0,SWA_BLOCK)
            slot=tl.load(source+req*STRIDE_SOURCE+cols,mask=cols<length,other=0)
            value=tl.load(mapping+slot.to(tl.int64),mask=cols<length,other=0)
            tl.store(sliding+row*STRIDE_SWA+cols,value,mask=cols<length)

def snapshot(lengths,cumulative,full,source,request,limit,seq,delta,page_size,
             sliding,mapping,*,swa_suffix_tokens,full_block=2048,swa_block=256,warps=4):
    """Pinned static token map, owned contiguous output columns and a bounded SWA suffix."""
    assert page_size==1 and swa_suffix_tokens>0 and limit>0
    assert all(t.is_contiguous() for t in (lengths,cumulative,request,seq,mapping))
    assert all(t.stride(1)==1 for t in (source,full,sliding))
    batch=seq.numel()
    full_tiles=triton.cdiv(limit,full_block)
    tiles=full_tiles+min(triton.cdiv(limit,swa_block),triton.cdiv(swa_suffix_tokens,swa_block)+1)
    _snapshot[(tiles,batch)](lengths,cumulative,full,source,request,seq,sliding,mapping,
        source.stride(0),full.stride(0),sliding.stride(0),limit,batch,triton.next_power_of_2(batch),
        delta,swa_suffix_tokens,full_block,swa_block,full_tiles,num_warps=warps)
