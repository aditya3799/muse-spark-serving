"""Research-only exact NVFP4 operand staging; no global scale folding."""
import torch
import triton
import triton.language as tl
@triton.jit
def dequant_kernel(W,S,Y,COUNT:tl.constexpr,BLOCK:tl.constexpr):
 off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);mask=off<COUNT
 byte=tl.load(W+off//2,mask=mask,other=0).to(tl.int32)
 code=(byte>>((off%2)*4))&15;mag=code&7
 q=tl.where(mag<2,mag*.5,tl.where(mag<4,1.+(mag-2)*.5,tl.where(mag<6,mag-2.,2.*(mag-4))))
 q=(q.to(tl.int32,bitcast=True)|((code&8)<<28)).to(tl.float32,bitcast=True)
 sb=tl.load(S+off//16,mask=mask,other=0).to(tl.int32);se=(sb>>3)&15;sm=sb&7
 normal=((se+120)<<23)|(sm<<20);sub=(sm.to(tl.float32)*(1./512)).to(tl.int32,bitcast=True)
 bits=tl.where(se==0,sub,normal);bits=tl.where((se==15)&(sm==7),0x7fc00000,bits)|(sb&128)<<24
 scale=bits.to(tl.float32,bitcast=True)
 tl.store(Y+off,(q*scale).to(tl.bfloat16),mask=mask)
@triton.jit
def scatter_kernel(X,IDS,Y,H:tl.constexpr,R:tl.constexpr,TOPK:tl.constexpr,EXPERT:tl.constexpr,BLOCK:tl.constexpr):
 off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=off<R*TOPK*H
 slot=off//H;eid=tl.load(IDS+slot,mask=valid,other=-1)
 mask=valid&(eid==EXPERT);v=tl.load(X+(slot//TOPK)*H+off%H,mask=mask,other=0)
 tl.store(Y+off,v,mask=mask)
def dequant(weight,scales,out):
 dequant_kernel[(triton.cdiv(out.numel(),4096),)](weight,scales.view(torch.uint8),out,out.numel(),4096)
def scatter(x,ids,out,expert):
 scatter_kernel[(triton.cdiv(out.numel(),1024),)](x,ids,out,x.shape[1],x.shape[0],ids.shape[1],expert,1024)
