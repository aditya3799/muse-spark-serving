"""Masked canonical output-column scatter; all costs belong inside the tiled call."""
import torch
import triton
import triton.language as tl
@triton.jit
def scatter_n_band_kernel(X,IDS,Y,R:tl.constexpr,N:tl.constexpr,T:tl.constexpr,OFFSET,E:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<R*T;row=i//T;col=i%T
 expert=tl.load(IDS+row,mask=valid,other=-1);active=valid&(expert>=0)&(expert<E)
 value=tl.load(X+i,mask=active,other=0);tl.store(Y+row*N+OFFSET+col,value,mask=active)
def scatter_n_band(x,ids,out,offset,experts=1):
 assert x.dtype==out.dtype==torch.bfloat16 and ids.dtype==torch.int32
 assert x.dim()==out.dim()==2 and x.is_cuda and ids.is_cuda and out.is_cuda and x.device==ids.device==out.device
 assert x.is_contiguous() and out.is_contiguous() and ids.is_contiguous()
 assert type(offset)==int and type(experts)==int and experts>0
 out_range=(out.data_ptr(),out.data_ptr()+out.numel()*out.element_size())
 for inp in (x,ids):
  inp_range=(inp.data_ptr(),inp.data_ptr()+inp.numel()*inp.element_size())
  assert out_range[1]<=inp_range[0] or inp_range[1]<=out_range[0]
 assert x.shape[0]==out.shape[0]==ids.numel() and 0<=offset<=out.shape[1]-x.shape[1]
 scatter_n_band_kernel[(triton.cdiv(x.numel(),512),)](x,ids,out,out.shape[0],out.shape[1],x.shape[1],offset,experts,512)
