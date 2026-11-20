"""Share original Humming packed weights with grouped SM90a. Change only 1-byte group scales."""
import torch
import triton
import triton.language as tl
from group_prep import _pack_scales
from exclusive_pack import Stage


def words_from_native(sf,experts,out,n,k):
    e=experts.numel()
    if (n%64 or k%64 or sf.numel()*sf.element_size()!=e*n*k//16
        or experts.dtype!=torch.int32 or out.dtype!=torch.int32 or out.numel()<e*n*k//64
        or not all(t.device==sf.device and t.is_contiguous() for t in (sf,experts,out))):
        raise ValueError("invalid group-16 input scales or uint32 destination")
    words=out.view(-1)[:e*n*k//64].view(e,k//64,n)
    _pack_scales[(n//64,k//64,e)](sf.view(torch.uint8),words,experts,n,k,64)
    return words


def install_scales(layer):
    if hasattr(layer,"_shared_scale_words"):
        return
    if hasattr(layer,"_exclusive_cutlass"):
        raise ValueError("requires the original Humming weight bank")
    infos={};staged={}
    for name in ("w13","w2"):
        m=layer.humming_metas[name]
        if (str(m.b_dtype)!="float4e2m1" or str(m.bs_dtype)!="float8e4m3" or m.weight_scale_group_size!=16
            or m.should_apply_bs_on_c or m.use_fused_e8m0_scale or m.use_packed_k_layout or m.has_bias or m.has_zero_point):
            raise ValueError("requires unreduced, unfused native NVFP4 and group-16 E4M3 scales")
        n,k,e=m.shape_n,m.shape_k,m.num_experts
        w,s=getattr(layer,m.weight_name),getattr(layer,m.weight_scale_name)
        if w.numel()*w.element_size()!=e*n*k//2:
            raise ValueError("invalid native weights")
        selected=torch.arange(e,device=w.device,dtype=torch.int32)
        out=torch.empty((e,k//64,n),device=w.device,dtype=torch.int32)
        words_from_native(s,selected,out,n,k)
        infos[name]=Stage(name,m.weight_name,m.weight_scale_name,tuple(w.shape),tuple(s.shape),
                          w.dtype,s.dtype,"wgmma" in str(m.mma_type).lower())
        staged[m.weight_scale_name]=torch.nn.Parameter(out,requires_grad=False)
    for key,value in staged.items():
        setattr(layer,key,value)
    layer._shared_scale_words=infos


@triton.jit
def _native_scales(src,active,dst,E:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                   B:tl.constexpr,WORKERS:tl.constexpr):
    task=tl.program_id(0)
    tiles=(N//B)*(K//16)
    end=tl.load(active+E)*tiles
    pos=tl.arange(0,B)
    while task<end:
        expert=tl.load(active+task//tiles).to(tl.int64)
        inside=task%tiles
        kg=inside//(N//B)
        v=inside%(N//B)*B+pos
        n=v//64*64+(v%8)*8+(v%64)//8
        word=tl.load(src+(expert*(K//64)+kg//4)*N+n).to(tl.uint32)
        tl.store(dst+(expert*(K//16)+kg)*N+v,(word>>((kg%4)*8))&255)
        task+=WORKERS


def native_from_words(packed,active,scratch,workers):
    e,k64,n=packed.shape
    b=min(128,n)
    if (n%b or n%64 or not packed.is_contiguous() or packed.dtype!=torch.int32 or
        scratch.dtype!=torch.uint8 or scratch.numel()<packed.numel()*4):
        raise ValueError("invalid native scale destination")
    dst=scratch.view(-1)[:packed.numel()*4].view(e,k64*4,n)
    _native_scales[(workers,)](packed,active,dst,e,n,k64*64,b,workers)
    return dst
