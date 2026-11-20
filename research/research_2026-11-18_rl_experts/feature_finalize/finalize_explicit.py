"""Two full-K feature shards; preserve the original expert-owner BF16 boundaries."""
import torch
import triton
import triton.language as tl


@triton.jit
def _joined(A,B,IDS,SCORES,NORM,SCALE,OWN,OUT,H:tl.constexpr,E:tl.constexpr):
    tok=tl.program_id(0)
    f=tl.arange(0,H)
    norm=tl.load(NORM+f).to(tl.float32)
    a0=tl.full((H,),0,tl.float32);a1=tl.full((H,),0,tl.float32)
    for j in tl.static_range(8):
        route=tok*8+j
        e=tl.load(IDS+route)
        if e>=0:
            # The original full-feature reduction visits the same lanes in the
            # same order; physical rank-major storage alone changes.
            base=route*(H//2)
            lo=tl.load(A+base+f,f<H//2,other=0.)
            hi=tl.load(B+base+(f-H//2),f>=H//2,other=0.)
            x=tl.where(f<H//2,lo,hi).to(tl.float32)
            post=tl.load(SCALE+e).to(tl.float32)
            x=(x*post).to(tl.bfloat16).to(tl.float32)
            v=x*norm
            inv=tl.rsqrt(tl.sum(v*v,axis=0)/H+1.e-8)
            z=v*inv
            s=tl.load(SCORES+route).to(tl.float32)
            if e<E:
                a0=tl.fma(z,s,a0)
            else:
                a1=tl.fma(z,s,a1)
    off=tok*H+f
    p=a0.to(tl.bfloat16);q=a1.to(tl.bfloat16)
    tl.store(OWN+2*tok*H+f,p);tl.store(OWN+(2*tok+1)*H+f,q)
    tl.store(OUT+off,p.to(tl.float32)+q.to(tl.float32))


@triton.jit
def _square(D,IDS,NORM,SCALE,PART,HALF:tl.constexpr):
    slot=tl.program_id(0)
    e=tl.load(IDS+slot)
    value=0.
    if e>=0:
        f=tl.arange(0,HALF)
        post=tl.load(SCALE+e).to(tl.float32)
        raw=tl.load(D+slot*HALF+f).to(tl.float32)
        x=(raw*post).to(tl.bfloat16).to(tl.float32)
        y=x*tl.load(NORM+f).to(tl.float32)
        value=tl.sum(y*y,axis=0)
    tl.store(PART+slot,value)


@triton.jit
def _shard_final(D,IDS,SCORES,NORM,SCALE,TOTAL,OWN,HALF:tl.constexpr,E:tl.constexpr,TILE:tl.constexpr):
    tok=tl.program_id(0)
    f=tl.program_id(1)*TILE+tl.arange(0,TILE)
    norm=tl.load(NORM+f).to(tl.float32)
    a0=tl.full((TILE,),0,tl.float32);a1=tl.full((TILE,),0,tl.float32)
    for j in tl.static_range(8):
        slot=tok*8+j;e=tl.load(IDS+slot)
        if e>=0:
            post=tl.load(SCALE+e).to(tl.float32)
            raw=tl.load(D+slot*HALF+f).to(tl.float32)
            x=(raw*post).to(tl.bfloat16).to(tl.float32)
            y=x*norm
            inv=tl.rsqrt(tl.load(TOTAL+slot)/(2*HALF)+1.e-8)
            score=tl.load(SCORES+slot).to(tl.float32)
            z=y*inv
            if e<E:
                a0=tl.fma(z,score,a0)
            else:
                a1=tl.fma(z,score,a1)
    # owner-major [2,M,F] directly feeds the two-rank all_to_all_single.
    m=tl.num_programs(0)
    tl.store(OWN+tok*HALF+f,a0);tl.store(OWN+(m+tok)*HALF+f,a1)


def joined(parts,ids,scores,norm,scale):
    a,b=parts
    m=ids.shape[0];h=norm.numel()
    assert len(parts)==2 and ids.shape==(m,8) and a.shape==b.shape==(8*m,h//2) and h in (512,1024,2048,4096)
    own=a.new_empty((m,2,h));out=a.new_empty((m,h))
    _joined[(m,)](a,b,ids,scores,norm,scale,own,out,h,scale.numel()//2,num_warps=8 if h>=4096 else 4,num_stages=2)
    return out,own.transpose(0,1)


def local_squares(down,ids,norm_half,scale):
    half=norm_half.numel()
    out=torch.empty(ids.numel(),device=down.device,dtype=torch.float32)
    _square[(ids.numel(),)](down,ids,norm_half,scale,out,half,num_warps=4,num_stages=2)
    return out


def local_owner_features(down,ids,scores,norm_half,scale,reduced_squares):
    half=norm_half.numel();m=ids.shape[0]
    out=down.new_empty((2,m,half))
    _shard_final[(m,half//256)](down,ids,scores,norm_half,scale,reduced_squares,out,half,scale.numel()//2,256,num_warps=4)
    return out


def paired_scalar_emulation(parts,ids,scores,norm,scale):
    norms=norm.chunk(2)
    sq=[local_squares(v,ids,n,scale) for v,n in zip(parts,norms)]
    totals=sq[0]+sq[1]
    data=[local_owner_features(v,ids,scores,n,scale,totals) for v,n in zip(parts,norms)]
    owner=torch.cat(data,dim=-1)
    return owner[0]+owner[1],owner
