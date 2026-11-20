"""Materialize a bounded BF16 expert slab directly from the incumbent's single Humming layout."""
import torch
import triton
import triton.language as tl

@triton.jit
def _decode(weights,scales,table,experts,out,N:tl.constexpr,K:tl.constexpr,
            TRANSPOSE_MINI:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    nn=tl.program_id(0)*BN+tl.arange(0,BN)
    kk=tl.program_id(1)*BK+tl.arange(0,BK)
    block=tl.program_id(2)
    e=tl.load(experts+block)
    # Native 64x16 weight tiles use one warp. Each lane owns one packed word per 16 output rows.
    lane=(nn[:,None]%8)*4+(kk[None,:]%8)//2
    word=(e.to(tl.int64)*(K//16)+kk[None,:]//16)*(N*2)
    word+=nn[:,None]//64*128+lane*4+(nn[:,None]%64)//16
    n_half=(nn[:,None]//8)%2
    k_half=(kk[None,:]//8)%2
    if TRANSPOSE_MINI:
        logical=(k_half*2+n_half)*2+kk[None,:]%2
    else:
        logical=(n_half*2+k_half)*2+kk[None,:]%2
    packed_nibble=(logical%2)*4+logical//2
    packed=tl.load(weights+word).to(tl.uint32)
    code=(packed>>(packed_nibble.to(tl.uint32)*4))&15
    # Group-16 input scales use the untransposed-feature application variant, an 8x8 permutation.
    idx=(e.to(tl.int64)*(K//16)+kk[None,:]//16)*N
    idx+=nn[:,None]//64*64+(nn[:,None]%8)*8+(nn[:,None]%64)//8
    scale=tl.load(scales+idx).to(tl.int32)
    value=tl.load(table+scale*16+code.to(tl.int32))
    tl.store(out+(block.to(tl.int64)*N+nn[:,None])*K+kk[None,:],value)


def make_table(device):
    code=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],device=device)
    scale=torch.arange(256,device=device).to(torch.uint8).view(torch.float8_e4m3fn).float()
    return (scale[:,None]*(code[None,:]/4)).bfloat16().contiguous()


def materialize(weights,scales,table,expert_ids,out,transpose):
    e,n,k=out.shape
    assert n%64==k%64==0 and e==expert_ids.numel()
    assert weights.is_contiguous() and scales.is_contiguous() and out.is_contiguous()
    _decode[(n//16,k//64,e)](weights.view(torch.int32),scales.view(torch.uint8),
                              table,expert_ids,out,n,k,transpose,16,64,num_warps=4)
    return out
