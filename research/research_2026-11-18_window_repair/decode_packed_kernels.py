"""CUDA packing helpers for the global draft top-k; new tensors have ordinary dense layout."""
import torch
import triton
import triton.language as tl

@triton.jit
def _pack_candidate_scores(vals,ids,packed,n,offset,WIDTH:tl.constexpr):
    x=tl.program_id(0)*WIDTH+tl.arange(0,WIDTH)
    v=tl.load(vals+x,x<n,other=0).to(tl.float32)
    bits=v.to(tl.uint32,bitcast=True).to(tl.uint64)
    index=tl.load(ids+x,x<n,other=0).to(tl.uint64)+offset
    tl.store(packed+x,((bits<<32)|index).to(tl.int64,bitcast=True),x<n)

@triton.jit
def _unpack_candidate_scores(packed,values,n,WIDTH:tl.constexpr):
    x=tl.program_id(0)*WIDTH+tl.arange(0,WIDTH)
    bits=tl.load(packed+x,x<n,other=0).to(tl.uint64,bitcast=True)
    tl.store(values+x,(bits>>32).to(tl.uint32).to(tl.float32,bitcast=True),x<n)

@triton.jit
def _gather_selected_candidate_ids(packed,selected,out,n,k:tl.constexpr,width:tl.constexpr,BLOCK:tl.constexpr):
    x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    idx=tl.load(selected+x,x<n,other=0)
    raw=tl.load(packed+(x//k)*width+idx,x<n,other=0)
    tl.store(out+x,raw&0xffffffff,x<n)

def pack_candidate_scores(values,ids,offset):
    values=values.contiguous()
    ids=ids.contiguous()
    out=torch.empty_like(ids,dtype=torch.int64)
    _pack_candidate_scores[(triton.cdiv(ids.numel(),256),)](values,ids,out,ids.numel(),offset,WIDTH=256)
    return out

def pick_packed_candidates(packed,k):
    packed=packed.contiguous()
    values=torch.empty_like(packed,dtype=torch.float32)
    _unpack_candidate_scores[(triton.cdiv(packed.numel(),256),)](packed,values,values.numel(),WIDTH=256)
    top,sel=torch.topk(values,k,dim=-1)
    out=torch.empty_like(sel,dtype=torch.int64)
    _gather_selected_candidate_ids[(triton.cdiv(out.numel(),256),)](packed,sel,out,out.numel(),k,packed.shape[-1],BLOCK=256)
    return out,top

