"""Prepare one ragged physical page list per logical request for both native FP8 variants."""
import torch
import triton
import triton.language as tl

@triton.jit
def _pages(src,request,prefix,reservations,fullmap,swamap,outer,inner,full,sliding,
           stride,LEFT:tl.constexpr,W:tl.constexpr,K:tl.constexpr):
    b=tl.program_id(0)
    cols=tl.program_id(1)*K+tl.arange(0,K)
    req=tl.load(request+b)
    committed=tl.load(prefix+b)
    old_swa=tl.minimum(committed,LEFT)
    stop=committed+W
    valid=cols<stop
    slot=tl.load(src+req.to(tl.int64)*stride+cols,mask=valid & (cols<committed),other=0)
    tail=tl.load(reservations+b*W+cols-committed,mask=valid & (cols>=committed),other=0)
    virt=tl.where(cols<committed,slot,tail)
    phys=tl.load(fullmap+virt.to(tl.int64),mask=valid,other=0)
    tp=tl.load(outer+b)
    tl.store(full+tp.to(tl.int64)+cols,phys,mask=valid)
    live_sw=valid & (cols>=committed-old_swa)
    swa_id=tl.load(swamap+virt.to(tl.int64),mask=live_sw,other=0)
    sp=tl.load(inner+b)
    tl.store(sliding+sp.to(tl.int64)+cols-(committed-old_swa),swa_id,mask=live_sw)

@triton.jit
def _pack_descriptor(depth,mask,out,N:tl.constexpr,K:tl.constexpr):
    i=tl.program_id(0)*K+tl.arange(0,K)
    d=tl.load(depth+i,i<N,other=0).to(tl.uint32);m=tl.load(mask+i,i<N,other=0).to(tl.uint32)
    tl.store(out+i,(d<<16)|m,i<N)


class SharedPages:
    def __init__(self,batch,capacity,width,left,device):
        if not 2<=width<=16 or left<width-1 or capacity<width:
            raise ValueError("Unsupported tree width, capacity, or sliding window")
        self.b=batch;self.w=width;self.cap=capacity;self.left=left
        i32=dict(device=device,dtype=torch.int32)
        self.full=torch.empty(batch*capacity,**i32)
        self.sliding=torch.empty(batch*min(capacity,left+width),**i32)
        self.qo=torch.arange(0,(batch+1)*width,width,**i32)
        self.last=torch.ones(batch,**i32)
        self.full_ptr=torch.zeros(batch+1,**i32);self.swa_ptr=torch.zeros_like(self.full_ptr)
        self.end=torch.empty(batch,**i32);self.swa_end=torch.empty_like(self.end)
        self.descriptor=torch.empty((batch,width),device=device,dtype=torch.uint32)

    def stage(self,table,req,prefix,slots,fullmap,swamap,depth,mask):
        if slots.shape!=(self.b,self.w) or mask.shape!=slots.shape or depth.shape!=slots.shape:
            raise ValueError("Unmatched reservation or topology")
        torch.add(prefix,self.w,out=self.end)
        torch.clamp(prefix,max=self.left,out=self.swa_end);self.swa_end.add_(self.w)
        torch.cumsum(self.end,0,dtype=torch.int32,out=self.full_ptr[1:])
        torch.cumsum(self.swa_end,0,dtype=torch.int32,out=self.swa_ptr[1:])
        _pages[(self.b,triton.cdiv(self.cap,256))](table,req,prefix,slots,fullmap,swamap,self.full_ptr,
            self.swa_ptr,self.full,self.sliding,table.stride(0),self.left,self.w,256,num_warps=4)
        _pack_descriptor[(triton.cdiv(self.b*self.w,128),)](depth,mask,self.descriptor,self.b*self.w,128)

