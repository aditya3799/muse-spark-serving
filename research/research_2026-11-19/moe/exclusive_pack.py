"""Transient restoration of only live Humming expert slices from one CUTLASS-order resident layout."""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from group_prep import convert


@triton.jit
def _present(ids, flags, S:tl.constexpr, E:tl.constexpr, B:tl.constexpr):
    e=tl.program_id(0)
    i=tl.program_id(1)*B+tl.arange(0,B)
    found=tl.sum((tl.load(ids+i,i<S,other=-1)==e).to(tl.int32),0)
    if found>0:
        tl.atomic_or(flags+e,1,sem="relaxed")


@triton.jit
def _compact(flags, active, E:tl.constexpr, B:tl.constexpr):
    e=tl.arange(0,B)
    hit=tl.load(flags+e,e<E,other=0)
    prefix=tl.cumsum(hit,0)
    tl.store(active+prefix-1,e,hit!=0)
    tl.store(active+E,tl.sum(hit,0))


def active_experts(ids,flags,active):
    e=flags.numel()
    if ids.dtype not in (torch.int32,torch.int64) or not ids.is_contiguous() or active.numel()!=e+1:
        raise ValueError("requires contiguous integer routes and one count slot")
    flags.zero_()
    if ids.numel():
        _present[(e,triton.cdiv(ids.numel(),256))](ids,flags,ids.numel(),e,256)
    _compact[(1,)](flags,active,e,triton.next_power_of_2(e))


@triton.jit
def _restore_native(q,sc,active,words,bytes_out,E:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                    WG:tl.constexpr,WORKERS:tl.constexpr):
    worker=tl.program_id(0)
    nc=N//64
    tiles=nc*(K//16)
    limit=tl.load(active+E)*tiles
    v=tl.arange(0,128)
    # A persistent grid reads exactly four packed CUTLASS bytes per original native word.
    task=worker
    while task<limit:
        expert=tl.load(active+task//tiles).to(tl.int64)
        within=task%tiles
        kb=within//nc
        nb=within%nc
        lane=v//4
        n=nb*64+(v%4)*16+lane//4
        kk=kb*8+lane%4
        word=tl.full((128,),0,tl.uint32)
        for j in tl.static_range(4):
            if WG:
                nh=j%2
                kh=j//2
            else:
                nh=j//2
                kh=j%2
            b=tl.load(q+(expert*N+n+nh*8)*(K//2)+kk+kh*4).to(tl.uint32)
            word |= ((b&15)<<(j*4))|((b&240)<<(j*4+12))
        tl.store(words+(expert*(K//16)+kb)*(2*N)+nb*128+v,word)
        physical=v%64
        original=nb*64+(physical%8)*8+physical//8
        x=tl.load(sc+(expert*(K//64)+kb//4)*N+original,v<64,other=0).to(tl.uint32)
        tl.store(bytes_out+(expert*(K//16)+kb)*N+nb*64+v,(x>>((kb%4)*8))&255,v<64)
        task+=WORKERS


def restore(codes, scale_words, active, scratch, transpose, workers):
    e,n,kb=codes.shape
    k=2*kb
    a=e*n*kb
    b=e*n*k//16
    if (codes.dtype!=torch.uint8 or scale_words.dtype!=torch.int32 or active.dtype!=torch.int32 or
        not all(x.is_contiguous() and x.device==codes.device for x in (codes,scale_words,active,scratch)) or
        n%64 or k%64 or scale_words.shape!=(e,k//64,n) or scratch.dtype!=torch.uint8 or
        scratch.numel()<a+b or active.numel()!=e+1 or workers<1):
        raise ValueError("invalid grouped canonical operands or native scratch")
    raw=scratch.view(-1)
    words=raw[:a].view(torch.int32).view(e,k//16,n*2)
    sf=raw[a:a+b].view(e,k//16,n)
    _restore_native[(workers,)](codes,scale_words,active,words,sf,e,n,k,transpose,workers,num_warps=4)
    return words,sf


@dataclass(frozen=True)
class Stage:
    name:str
    weight_name:str
    scale_name:str
    shape_w:tuple
    shape_s:tuple
    dtype_w:torch.dtype
    dtype_s:torch.dtype
    transpose:bool


def install_layer(layer):
    """Replace one prepared inference layer's native weight/scale Parameters, before capture."""
    if hasattr(layer,"_exclusive_cutlass"):
        return
    stages={}
    # Validate every stage and dispatch format before changing the resident representation.
    for name in ("w13","w2"):
        m=layer.humming_metas[name]
        w,s=getattr(layer,m.weight_name),getattr(layer,m.weight_scale_name)
        if (str(m.b_dtype)!="float4e2m1" or str(m.bs_dtype)!="float8e4m3" or
            m.weight_scale_group_size!=16 or m.has_zero_point or m.has_bias or
            m.should_apply_bs_on_c or m.use_fused_e8m0_scale or m.use_packed_k_layout or
            m.shape_n%64 or m.shape_k%64 or w.numel()*w.element_size()!=m.num_experts*m.shape_n*m.shape_k//2 or
            s.numel()*s.element_size()!=m.num_experts*m.shape_n*m.shape_k//16):
            raise ValueError("unsupported prepared layer format")
        stages[name]=Stage(name,m.weight_name,m.weight_scale_name,tuple(w.shape),tuple(s.shape),
                           w.dtype,s.dtype,"wgmma" in str(m.mma_type).lower())
    # Convert both before publishing the new parameters. Startup transient is one layer, no old-bank references retained.
    replacements={}
    for name,info in stages.items():
        m=layer.humming_metas[name]
        w,s=getattr(layer,info.weight_name),getattr(layer,info.scale_name)
        chosen=torch.arange(m.num_experts,device=w.device,dtype=torch.int32)
        dst=torch.empty(m.num_experts*m.shape_n*m.shape_k*9//16,device=w.device,dtype=torch.uint8)
        q,p=convert(w,s,chosen,dst,m.shape_n,m.shape_k,info.transpose)
        replacements[info.weight_name]=torch.nn.Parameter(q,requires_grad=False)
        replacements[info.scale_name]=torch.nn.Parameter(p,requires_grad=False)
    for key,value in replacements.items():
        setattr(layer,key,value)
    layer._exclusive_cutlass=stages


class NativeFallback:
    """Allocate once per serving stream. The original helper retains its native FP32/tiling decisions."""
    def __init__(self,nbytes,experts,device,workers=0):
        self.scratch=torch.zeros(nbytes,device=device,dtype=torch.uint8)
        self.flags=torch.empty(experts,device=device,dtype=torch.int32)
        self.active=torch.empty(experts+1,device=device,dtype=torch.int32)
        self.workers=workers or 2*torch.cuda.get_device_properties(device).multi_processor_count
        self.stream=torch.cuda.current_stream(device) if self.scratch.is_cuda else None

    def view(self,layer,ids,*,lazy=False):
        if (self.flags.numel()!=layer.humming_metas["w2"].num_experts or
            ids.device!=self.scratch.device or
            (self.stream is not None and torch.cuda.current_stream(ids.device)!=self.stream)):
            raise ValueError("exclusive fallback requires its owning stream and number of experts")
        if not lazy:
            active_experts(ids,self.flags,self.active)
        return _StageView(layer,self,ids if lazy else None)


class _StageView:
    """A new view per logical forward captures exactly one restore per distinct contiguous projection."""
    def __init__(self,layer,arena,pending):
        self._pending=pending
        self._layer=layer
        self._arena=arena
        self._shared=hasattr(layer,"_shared_scale_words")
        stages=layer._shared_scale_words if self._shared else layer._exclusive_cutlass
        self._names={key:stage for stage in stages.values() for key in (stage.weight_name,stage.scale_name)}
        self._prepared=None
        self._native={}

    def __getattr__(self,key):
        info=self._names.get(key)
        if info is None:
            return getattr(self._layer,key)
        if self._shared and key==info.weight_name:
            return getattr(self._layer,key)
        if self._pending is not None:
            active_experts(self._pending,self._arena.flags,self._arena.active)
            self._pending=None
        if self._prepared!=info.name:
            q=getattr(self._layer,info.weight_name)
            p=getattr(self._layer,info.scale_name)
            if self._shared:
                from shared_scales import native_from_words
                s=native_from_words(p,self._arena.active,self._arena.scratch,self._arena.workers)
                w=q
            else:
                w,s=restore(q,p,self._arena.active,self._arena.scratch,info.transpose,self._arena.workers)
            self._native={info.weight_name:w.view(info.dtype_w).reshape(info.shape_w),
                          info.scale_name:s.view(info.dtype_s).reshape(info.shape_s)}
            self._prepared=info.name
        return self._native[key]
