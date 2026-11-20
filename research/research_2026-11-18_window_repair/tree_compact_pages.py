"""Bounded contiguous logical SWA table. Absolute positions remain attached to rotary K/Q."""
import triton
import triton.language as tl


@triton.jit
def _compact_swa(dst,prefix,branches,lens,anc,mapping,lengths,
                 stride,pstride,astride,LEFT:tl.constexpr,WIDTH:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    req=row//WIDTH
    p=tl.load(lens+req)
    end=p+tl.load(anc+row)
    count=tl.minimum(end,LEFT+1)
    col=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    j=end-count+col
    valid=col<count
    a=tl.load(prefix+req*pstride+j,mask=valid & (j<p),other=0)
    b=tl.load(branches+row*astride+j-p,mask=valid & (j>=p),other=0)
    virtual=tl.where(j<p,a,b)
    physical=tl.load(mapping+virtual.to(tl.int64),mask=valid,other=0)
    tl.store(dst+row.to(tl.int64)*stride+col,physical,mask=valid)
    if tl.program_id(1)==0:
        tl.store(lengths+row,count)


def fill_compact_swa(dst,prefix,branches,lens,anc,mapping,lengths,window,nodes):
    _compact_swa[(dst.shape[0],triton.cdiv(min(window+1,dst.shape[1]),256))](
        dst,prefix,branches,lens,anc,mapping,lengths,dst.stride(0),prefix.stride(0),
        branches.stride(0),window,nodes,256,num_warps=4)

