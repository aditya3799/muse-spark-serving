"""Prefix-boundary GPU operations only. Full draft8 remains unchanged."""
import torch
from dflash_prefix_research import spans_overlap
import triton
import triton.language as tl

@triton.jit
def _pack4(A,B,C,D,OA,OB,OC,OD,N:tl.constexpr,K:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=i<N
    src=(i//K)*8+i%K
    tl.store(OA+i,tl.load(A+src,valid,other=0),valid)
    tl.store(OB+i,tl.load(B+src,valid,other=0),valid)
    tl.store(OC+i,tl.load(C+src,valid,other=0),valid)
    tl.store(OD+i,tl.load(D+src,valid,other=0),valid)

def pack_prefix_into(full,packed,width):
    # full: token IDs, absolute positions, target pool IDs, draft pool IDs.
    # Separate physical spaces must never be interchanged.
    if width not in (1,2,4,6,8) or len(full)!=4 or len(packed)!=4:
        raise ValueError('Expected four full8 inputs and supported width')
    bs=full[0].shape[0]
    for x,y in zip(full,packed):
        if x.shape!=(bs,8) or y.shape!=(bs,width):
            raise ValueError('Invalid prefix shape')
        if x.dtype!=torch.int64 or y.dtype!=torch.int64:
            raise ValueError('Protocol diagnostic supports int64 only')
        if not x.is_cuda or y.device!=x.device or x.device!=full[0].device:
            raise ValueError('All buffers must share a CUDA device')
        if not x.is_contiguous() or not y.is_contiguous():
            raise ValueError('Buffers must be contiguous')
    # Contiguous spans, including views with unequal data pointers. Inputs may
    # share read-only storage; every output must be disjoint from all inputs
    # and from every other output.
    for i,y in enumerate(packed):
        for x in list(full)+list(packed[:i]):
            if spans_overlap(x.data_ptr(),x.numel()*x.element_size(),
                             y.data_ptr(),y.numel()*y.element_size()):
                raise ValueError('Prefix output overlaps an input or another output')
    if bs:
        _pack4[(triton.cdiv(bs*width,128),)](*full,*packed,bs*width,width,128)
