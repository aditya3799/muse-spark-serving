"""Lossless packed maxbits/int64-ID transport and first-rank selection."""
import triton
import triton.language as tl

@triton.jit
def pack_max_id(MAX,IDS,PACKED,N:tl.constexpr,BLOCK:tl.constexpr,COMPACT:tl.constexpr=False):
 ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 value=tl.load(MAX+ix,ix<N,other=0)
 if MAX.dtype.element_ty==tl.float32:
  bits=value.to(tl.uint32,bitcast=True).to(tl.int64)
 else:
  bits=value.to(tl.uint16,bitcast=True).to(tl.int64)
 ids=tl.load(IDS+ix,ix<N,other=0)
 if COMPACT:
  word=(bits.to(tl.uint64)<<32)|ids.to(tl.uint32).to(tl.uint64)
  tl.store(PACKED+ix,word,ix<N)
 else:
  tl.store(PACKED+2*ix,bits,ix<N)
  tl.store(PACKED+2*ix+1,ids,ix<N)

@triton.jit
def select_first_rank(PACKED,OUT,N:tl.constexpr,TP:tl.constexpr,RT:tl.constexpr,VALUE_DTYPE:tl.constexpr,BLOCK:tl.constexpr,COMPACT:tl.constexpr=False):
 token=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 rank=tl.arange(0,RT)
 mask=(rank[:,None]<TP)&(token[None,:]<N)
 if COMPACT:
  words=tl.load(PACKED+rank[:,None]*N+token[None,:],mask,other=0).to(tl.uint64)
  bits=words>>32
 else:
  bits=tl.load(PACKED+rank[:,None]*(2*N)+2*token[None,:],mask,other=0)
 if VALUE_DTYPE==tl.float32:
  values=bits.to(tl.uint32).to(tl.float32,bitcast=True)
 else:
  values=bits.to(tl.uint16).to(VALUE_DTYPE,bitcast=True).to(tl.float32)
 nan=(values!=values)&mask
 first_nan=tl.min(tl.where(nan,rank[:,None],TP),axis=0)
 maximum=tl.max(tl.where(mask&~nan,values,float('-inf')),axis=0)
 best=tl.min(tl.where(mask&(values==maximum[None,:]),rank[:,None],TP),axis=0)
 best=tl.where(first_nan<TP,first_nan,best)
 if COMPACT:
  selected=tl.load(PACKED+best*N+token,token<N,other=0).to(tl.uint64)&0xffffffff
 else:
  selected=tl.load(PACKED+best*(2*N)+2*token+1,token<N,other=0)
 tl.store(OUT+token,selected,token<N)
