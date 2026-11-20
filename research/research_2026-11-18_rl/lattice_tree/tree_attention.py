"""Page-one GQA verification with a shared prefix and a per-node ancestor mask, for local replay.

This benchmark reads physical FP8 E4M3FN bytes into exact FP16 values on Ampere.
Hopper FA3 has its own FP8 softmax and epilogue. Local comparisons use an independent
dense reference and cannot establish parity with the serving binary.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _part(q,k,v,page,slots,lens,depth,bits,row_ids,lut,partials,maxima,weights,
          page_stride, scale, SPLITS:tl.constexpr, CHUNK:tl.constexpr,
          LEFT:tl.constexpr, N:tl.constexpr, G:tl.constexpr, H:tl.constexpr,
          D:tl.constexpr, M:tl.constexpr, BK:tl.constexpr, TREE:tl.constexpr, DIRECT:tl.constexpr):
    group=tl.program_id(0)
    s=tl.program_id(1)
    b=group//H
    h=group%H
    r=tl.arange(0,M)
    d=tl.arange(0,D)
    t=tl.arange(0,BK)
    node=r//G
    query_head=h*G+r%G
    nmask=node<N
    if DIRECT:
        physical_row=tl.load(row_ids+b)
    else:
        physical_row=b
    qbyte=tl.load(q+((b*N+node[:,None])*H*G+query_head[:,None])*D+d[None,:],mask=nmask[:,None],other=0)
    qq=tl.load(lut+qbyte.to(tl.int32)).to(tl.float16)
    seq=tl.load(lens+b)
    if TREE:
        relative=tl.load(depth+b*N+node,mask=nmask,other=0)
        ancestry=tl.load(bits+b*N+node,mask=nmask,other=0).to(tl.uint32)
    else:
        relative=node
    p=tl.full((M,),-float("inf"),tl.float32)
    norm=tl.full((M,),0.,tl.float32)
    acc=tl.full((M,D),0.,tl.float32)
    lo=tl.full((),0,tl.int32)
    if LEFT>0:
        lo=tl.maximum(0,seq-LEFT)
    # Last split contains just the new verification nodes. All preceding splits reuse the common prefix.
    rounds=tl.where(s==SPLITS,1,tl.maximum(0,tl.minimum(CHUNK,seq-lo-s*CHUNK)+BK-1)//BK)
    for j in range(rounds):
        col=lo+s*CHUNK+j*BK+t
        if s==SPLITS:
            valid=t<N
            physical=tl.load(slots+b*N+t,mask=valid,other=0)
            if TREE:
                allowed=((ancestry[:,None]>>(t[None,:]%32).to(tl.uint32))&1)!=0
                if LEFT>0:
                    key_depth=tl.load(depth+b*N+t,mask=valid,other=0)
                    allowed=allowed & (key_depth[None,:]>=relative[:,None]-LEFT)
            else:
                allowed=t[None,:]<=node[:,None]
                if LEFT>0:allowed=allowed & (t[None,:]>=relative[:,None]-LEFT)
            visible=allowed & valid[None,:] & nmask[:,None]
        else:
            valid=col<seq
            physical=tl.load(page+physical_row.to(tl.int64)*page_stride+col,mask=valid,other=0)
            visible=valid[None,:] & nmask[:,None]
            if LEFT>0:visible=visible & (col[None,:]>=seq+relative[:,None]-LEFT)
        off=((physical.to(tl.int64)*H+h)[:,None]*D+d[None,:])
        kb=tl.load(k+off,mask=valid[:,None],other=0).to(tl.int32)
        vb=tl.load(v+off,mask=valid[:,None],other=0).to(tl.int32)
        kk=tl.load(lut+kb).to(tl.float16)
        vv=tl.load(lut+vb).to(tl.float16)
        logits=tl.dot(qq,tl.trans(kk))*scale
        logits=tl.where(visible,logits,-float("inf"))
        new_p=tl.maximum(p,tl.max(logits,1))
        stable=tl.where(new_p==-float("inf"),0.,new_p)
        renorm=tl.exp(p-stable)
        probs=tl.exp(logits-stable[:,None])
        norm=norm*renorm+tl.sum(probs,1)
        acc=acc*renorm[:,None]+tl.dot(probs.to(tl.float16),vv)
        p=new_p
    base=(group*(SPLITS+1)+s)*M+r
    tl.store(partials+base[:,None]*D+d[None,:],acc)
    tl.store(maxima+base,p)
    tl.store(weights+base,norm)


@triton.jit
def _finish(partials,maxima,weights,out,SPLITS:tl.constexpr,NS:tl.constexpr,
            M:tl.constexpr,N:tl.constexpr,G:tl.constexpr,H:tl.constexpr,D:tl.constexpr):
    group=tl.program_id(0)
    r=tl.program_id(1)
    s=tl.arange(0,NS)
    d=tl.arange(0,D)
    base=(group*(SPLITS+1)+s)*M+r
    valid=s<=SPLITS
    p=tl.load(maxima+base,mask=valid,other=-float("inf"))
    w=tl.load(weights+base,mask=valid,other=0.)
    factor=tl.exp(p-tl.max(p,0))
    total=tl.sum(factor*w,0)
    chunks=tl.load(partials+base[:,None]*D+d[None,:],mask=valid[:,None],other=0.)
    average=tl.sum(chunks*factor[:,None],0)/total
    node=r//G
    head=(group%H)*G+r%G
    tl.store(out+(((group//H)*N+node)*H*G+head)*D+d,average)


class AttentionReplay:
    def __init__(self,b,n,ctx,device,window=0,chunk=512):
        self.n,self.window,self.chunk=n,window,chunk
        self.h,self.g,self.d=2,8,64
        self.m=triton.next_power_of_2(n*self.g)
        self.s=triton.cdiv(min(ctx,window) if window else ctx,chunk)
        geom=(b*self.h,self.s+1,self.m)
        self.part=torch.empty((*geom,self.d),device=device,dtype=torch.float32)
        self.peak=torch.empty(geom,device=device,dtype=torch.float32)
        self.norm=torch.empty_like(self.peak)
        self.out=torch.empty((b,n,self.h*self.g,self.d),device=device,dtype=torch.float32)
        self.lut=torch.arange(256,device=device,dtype=torch.uint8).view(torch.float8_e4m3fn).float()

    def run(self,q,k,v,pages,slots,lens,depth,bits,tree,row_ids=None):
        assert q.dtype==k.dtype==v.dtype==torch.uint8 and all(x.is_contiguous() for x in (q,k,v,slots))
        b=q.shape[0]
        _part[(b*self.h,self.s+1)](q,k,v,pages,slots,lens,depth,bits,row_ids,self.lut,self.part,
            self.peak,self.norm,pages.stride(0),self.d**-.5,self.s,self.chunk,self.window,self.n,
            self.g,self.h,self.d,self.m,32,tree,row_ids is not None,num_warps=4)
        _finish[(b*self.h,self.n*self.g)](self.part,self.peak,self.norm,self.out,self.s,
            triton.next_power_of_2(self.s+1),self.m,self.n,self.g,self.h,self.d,num_warps=4)
        return self.out
