"""Compute the original softcapped vocabulary winner with independent CTAs for wide rows."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _tile(logits, parts, stride, ncols, cap, scale, tiles:tl.constexpr, WIDTH:tl.constexpr):
    tile=tl.program_id(0)
    row=tl.program_id(1).to(tl.int64)
    cols=tile*WIDTH+tl.arange(0,WIDTH)
    keep=cols<ncols
    x=tl.load(logits+row*stride+cols,mask=keep,other=0.)
    x=tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[x,scale], dtype=tl.float32,is_pure=True,pack=1)
    x=libdevice.tanh(x/cap)*cap
    x=tl.where(keep,x,-float("inf"))
    val,which=tl.max(x,axis=0,return_indices=True,return_indices_tie_break_left=True)
    offset=(row*tiles+tile)*2
    tl.store(parts+offset,val)
    tl.store(parts+offset+1,(which+tile*WIDTH).to(tl.float32))


@triton.jit
def _first_winner(parts, out, token_offset, tiles:tl.constexpr, BLOCK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    t=tl.arange(0,BLOCK)
    v=tl.load(parts+(row*tiles+t)*2,mask=t<tiles,other=-float("inf"))
    # The parent updates only on '>'. NaN and -inf tile winners never replace its initial value.
    v=tl.where(v==v,v,-float("inf"))
    best,first=tl.max(v,axis=0,return_indices=True,return_indices_tie_break_left=True)
    idx=tl.load(parts+(row*tiles+first)*2+1)
    idx=tl.where(best>-float("inf"),idx,0.)
    tl.store(out+row*2,best)
    tl.store(out+row*2+1,idx+token_offset)


def parallel_softcap_max(logits,cap,ncols,token_offset,tile=4096,scale=1.):
    assert logits.dtype==torch.float32 and logits.stride(1)==1 and token_offset+logits.shape[1]<2**24
    rows=logits.shape[0]
    tiles=max(1,triton.cdiv(ncols,tile))
    parts=torch.empty(rows,tiles,2,dtype=logits.dtype,device=logits.device)
    out=torch.empty(rows,2,dtype=logits.dtype,device=logits.device)
    if rows:
        _tile[(tiles,rows)](logits,parts,logits.stride(0),ncols,cap,scale,tiles,tile,num_warps=8)
        _first_winner[(rows,)](parts,out,token_offset,tiles,triton.next_power_of_2(tiles),num_warps=1)
    return out