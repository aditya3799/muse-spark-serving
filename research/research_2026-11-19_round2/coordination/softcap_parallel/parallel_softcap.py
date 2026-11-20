import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def _partial(LOGITS, PARTIAL, STRIDE, WIDTH, CAP, TILES:tl.constexpr, BLOCK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    tile=tl.program_id(1)
    start=tile*BLOCK
    cols=start+tl.arange(0,BLOCK)
    mask=cols<WIDTH
    x=tl.load(LOGITS+row*STRIDE+cols,mask=mask,other=0.0)
    x=x/CAP
    x=libdevice.tanh(x)
    x=x*CAP
    x=tl.where(mask,x,-float("inf"))
    block_val,block_idx=tl.max(x,axis=0,return_indices=True,return_indices_tie_break_left=True)
    tl.store(PARTIAL+2*(row*TILES+tile),block_val)
    tl.store(PARTIAL+2*(row*TILES+tile)+1,(start+block_idx).to(tl.float32))

@triton.jit
def _finish(PARTIAL,OUT,OFFSET,TILES:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    best_val=-float("inf")
    best_idx=0
    for tile in range(TILES):
        block_val=tl.load(PARTIAL+2*(row*TILES+tile))
        block_idx=tl.load(PARTIAL+2*(row*TILES+tile)+1).to(tl.int32)
        take=block_val>best_val
        best_idx=tl.where(take,block_idx,best_idx)
        best_val=tl.where(take,block_val,best_val)
    tl.store(OUT+row*2,best_val)
    tl.store(OUT+row*2+1,(best_idx+OFFSET).to(tl.float32))

def parallel_softcap(logits,cap,valid_width,col_offset):
    assert logits.dtype==torch.float32 and logits.stride(1)==1
    assert col_offset>=0 and logits.shape[1]<2**24
    assert col_offset+logits.shape[1]<2**24
    assert 0<=valid_width<=logits.shape[1]
    tiles=max(1,triton.cdiv(valid_width,4096))
    with torch.cuda.device(logits.device):
        scratch=torch.empty((logits.shape[0],tiles,2),dtype=torch.float32,device=logits.device)
        out=torch.empty((logits.shape[0],2),dtype=torch.float32,device=logits.device)
        _partial[(logits.shape[0],tiles)](logits,scratch,logits.stride(0),valid_width,cap,tiles,4096,num_warps=8)
        _finish[(logits.shape[0],)](scratch,out,col_offset,tiles,num_warps=4)
    return out
