"""Match Humming's K-warp traversal and final upper-half-to-lower-half reduction."""
import torch
import triton
import triton.language as tl

@triton.jit
def _partials(X,W,Z,ROWS:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BK:tl.constexpr,WK:tl.constexpr):
    m=tl.program_id(0)*16+tl.arange(0,16)
    n=tl.program_id(1)*32+tl.arange(0,32)
    worker=tl.program_id(2)
    acc=tl.full((16,32),0,tl.float32)
    for t in range(K//BK):
        for local in tl.static_range(WK//16):
            k=t*BK+worker*WK+local*16+tl.arange(0,16)
            a=tl.load(X+m[:,None]*K+k[None,:],mask=m[:,None]<ROWS,other=0)
            b=tl.load(W+n[:,None]*K+k[None,:],mask=n[:,None]<N,other=0)
            acc=tl.dot(a,tl.trans(b),acc)
    tl.store(Z+(worker*ROWS+m[:,None])*N+n[None,:],acc,mask=(m[:,None]<ROWS)&(n[None,:]<N))

def ordered(x,w,bk,wk):
    m,k=x.shape;n=w.shape[0]
    assert x.is_contiguous() and w.is_contiguous() and bk%wk==wk%16==k%bk==0
    splits=bk//wk
    assert splits>0 and not splits&(splits-1)
    parts=torch.empty((splits,m,n),device=x.device,dtype=torch.float32)
    _partials[(triton.cdiv(m,16),triton.cdiv(n,32),splits)](x,w,parts,m,n,k,bk,wk,num_warps=4)
    while splits>1:
        splits//=2
        parts[:splits].add_(parts[splits:2*splits])
    return (parts[0]*4).bfloat16()
