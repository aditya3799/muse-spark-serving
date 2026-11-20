"""GPU-only fixed-capacity expert compaction, stable original-slot inverse map."""
import torch,triton
import triton.language as tl
@triton.jit
def _hist(ID,P,S:tl.constexpr,E:tl.constexpr,B:tl.constexpr,BINS:tl.constexpr):
 chunk=tl.program_id(0);i=chunk*B+tl.arange(0,B);v=tl.load(ID+i,i<S,other=-1);v=tl.where((v>=0)&(v<E)&(i<S),v,BINS-1);h=tl.histogram(v,BINS);e=tl.arange(0,BINS);tl.store(P+chunk*E+e,h,e<E)
@triton.jit
def _prefix(P,C,L,H,E:tl.constexpr,CHUNKS:tl.constexpr,PC:tl.constexpr,PE:tl.constexpr,CUTOFF:tl.constexpr):
 ch=tl.arange(0,PC);e=tl.arange(0,PE);v=tl.load(P+ch[:,None]*E+e[None,:],(ch[:,None]<CHUNKS)&(e[None,:]<E),other=0);counts=tl.sum(v,0);lc=tl.where(counts<=CUTOFF,counts,0);hc=counts-lc;ls=tl.cumsum(lc);hs=tl.cumsum(hc);total_l=tl.sum(lc);tl.store(C+e,counts,e<E);tl.store(L+e,ls-lc,e<E);tl.store(H+e,total_l+hs-hc,e<E);tl.store(L+E,total_l);tl.store(H+E,total_l+tl.sum(hc))
@triton.jit
def _init_maps(MAP,INV,EX,S:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);tl.store(MAP+i,-1,i<S);tl.store(INV+i,-1,i<S);tl.store(EX+i,-1,i<S)
@triton.jit
def _map(ID,P,C,L,H,MAP,INV,EX,S:tl.constexpr,E:tl.constexpr,CHUNKS:tl.constexpr,PC:tl.constexpr,B:tl.constexpr,CUTOFF:tl.constexpr):
 expert=tl.program_id(0);chunk=tl.program_id(1);i=chunk*B+tl.arange(0,B);v=tl.load(ID+i,i<S,other=-1);valid=(i<S)&(v==expert);rank=tl.cumsum(valid.to(tl.int32))-1;ch=tl.arange(0,PC);before=tl.sum(tl.load(P+ch*E+expert,ch<chunk,other=0));count=tl.load(C+expert);base=tl.where(count<=CUTOFF,tl.load(L+expert),tl.load(H+expert));pos=base+before+rank;tl.store(MAP+pos,i,valid);tl.store(INV+i,pos,valid);tl.store(EX+pos,expert,valid)
@triton.jit
def _gather(X,MAP,H,Y,S:tl.constexpr,K:tl.constexpr,TOPK:tl.constexpr,E:tl.constexpr,B:tl.constexpr):
 ix=tl.program_id(0)*B+tl.arange(0,B);row=ix//K;col=ix%K;total=tl.load(H+E);slot=tl.load(MAP+row,(row<S)&(row<total),other=0);v=tl.load(X+(slot//TOPK)*K+col,(row<S)&(row<total),other=0);tl.store(Y+ix,v,(row<S)&(row<total))
@triton.jit
def _scatter(X,INV,Y,S:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
 ix=tl.program_id(0)*B+tl.arange(0,B);slot=ix//K;col=ix%K;row=tl.load(INV+slot,slot<S,other=-1);v=tl.load(X+row*K+col,(slot<S)&(row>=0),other=0);tl.store(Y+ix,v,(slot<S)&(row>=0))
class CompactRoutes:
 def __init__(self,ids,experts,cutoff):
  if ids.dtype!=torch.int32 or not ids.is_cuda or not ids.is_contiguous() or ids.ndim!=2:raise ValueError('Contiguous CUDA int32 routes required')
  self.shape=tuple(ids.shape);self.S=ids.numel();self.E=experts;self.cutoff=cutoff;self.B=1024;self.chunks=triton.cdiv(self.S,self.B);dev=ids.device
  if not 0<self.S<=131072 or not 0<experts<=256 or cutoff<0:raise ValueError('Unsupported routecapacity/expertcount')
  self.partial=torch.empty((self.chunks,experts),device=dev,dtype=torch.int32);self.counts=torch.empty((experts,),device=dev,dtype=torch.int32);self.light=torch.empty((experts+1,),device=dev,dtype=torch.int32);self.heavy=torch.empty_like(self.light);self.map=torch.empty((self.S,),device=dev,dtype=torch.int32);self.inverse=torch.empty_like(self.map);self.experts=torch.empty_like(self.map)
 def update(self,ids):
  if tuple(ids.shape)!=self.shape or ids.device!=self.map.device or ids.dtype!=torch.int32 or not ids.is_contiguous():raise ValueError('Stable route metadata required')
  S,E,B=self.S,self.E,self.B;_hist[(self.chunks,)](ids,self.partial,S,E,B,triton.next_power_of_2(E+1));_prefix[(1,)](self.partial,self.counts,self.light,self.heavy,E,self.chunks,triton.next_power_of_2(self.chunks),triton.next_power_of_2(E),self.cutoff,num_warps=4);_init_maps[(triton.cdiv(S,256),)](self.map,self.inverse,self.experts,S,256);_map[(E,self.chunks)](ids,self.partial,self.counts,self.light,self.heavy,self.map,self.inverse,self.experts,S,E,self.chunks,triton.next_power_of_2(self.chunks),B,self.cutoff,num_warps=4)
 def gather(self,x,out):
  if x.ndim!=2 or out.ndim!=2 or x.device!=self.map.device or out.device!=x.device or x.untyped_storage().data_ptr()==out.untyped_storage().data_ptr():raise ValueError("Gather requires disjoint same-device matrices")
  if x.dtype!=torch.bfloat16 or out.dtype!=x.dtype or not x.is_contiguous() or not out.is_contiguous() or tuple(out.shape)!=(self.S,x.shape[1]) or x.shape[0]!=self.shape[0]:raise ValueError('Gather geometry mismatch')
  _gather[(triton.cdiv(out.numel(),1024),)](x,self.map,self.heavy,out,self.S,x.shape[1],self.shape[1],self.E,1024)
 def scatter(self,x,out):
  if x.ndim!=2 or out.ndim!=2 or x.device!=self.map.device or out.device!=x.device or x.untyped_storage().data_ptr()==out.untyped_storage().data_ptr():raise ValueError("Scatter requires disjoint same-device matrices")
  if x.shape!=out.shape or x.shape[0]!=self.S or x.dtype!=torch.bfloat16 or out.dtype!=x.dtype or not x.is_contiguous() or not out.is_contiguous():raise ValueError('Scatter geometry mismatch')
  _scatter[(triton.cdiv(out.numel(),1024),)](x,self.inverse,out,self.S,x.shape[1],1024)
