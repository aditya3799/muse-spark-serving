"""Stable suffix-anchored per-expert grouped Humming, with device-only route maps."""
import torch
import triton
import triton.language as tl
from humming.layer import HummingMethod


@triton.jit
def _local_ids(ids, partial, ranks, ROUTES:tl.constexpr, BLOCK:tl.constexpr):
    expert = tl.program_id(0)
    chunk = tl.program_id(1)
    pos = chunk * BLOCK + tl.arange(0, BLOCK)
    key = tl.load(ids+pos, pos<ROUTES, other=-1)
    hit = key == expert
    rank = tl.cumsum(hit.to(tl.int32), 0) - 1
    tl.store(ranks+pos, rank, hit)
    tl.store(partial+chunk*32+expert, tl.sum(hit.to(tl.int32),0))


@triton.jit
def _offsets(prefix, offsets, base, CHUNKS:tl.constexpr, LIGHT:tl.constexpr, HEAVY:tl.constexpr, CUTOFF:tl.constexpr):
    experts = tl.arange(0,32)
    counts = tl.load(prefix+(CHUNKS-1)*32+experts)
    light = counts <= CUTOFF
    nl = tl.where(light, counts, 0)
    nh = tl.where(light, 0, counts)
    a = LIGHT - tl.sum(nl,0) + tl.cumsum(nl,0) - nl
    b = HEAVY - tl.sum(nh,0) + tl.cumsum(nh,0) - nh
    tl.store(offsets+experts,a)
    tl.store(offsets+32,LIGHT)
    tl.store(offsets+33+experts,b)
    tl.store(offsets+65,HEAVY)
    tl.store(base+experts,tl.where(light,a,LIGHT+b))


@triton.jit
def _complete(ids, ranks, prefix, base, ROUTES:tl.constexpr, BLOCK:tl.constexpr):
    ch = tl.program_id(0)
    pos = ch*BLOCK+tl.arange(0,BLOCK)
    e = tl.load(ids+pos,pos<ROUTES,other=-1)
    live=e>=0
    rank=tl.load(ranks+pos,live,other=0)
    counts=tl.load(prefix+(ch-1)*32+e,live & (ch>0),other=0)
    offset=tl.load(base+e,live,other=0)
    tl.store(ranks+pos,tl.where(live,rank+counts+offset,-1),pos<ROUTES)


@triton.jit
def _copy(x,y,ranks,H:tl.constexpr,PACK:tl.constexpr,TILE:tl.constexpr):
    route=tl.program_id(0)
    placed=tl.load(ranks+route)
    if placed>=0:
        j=tl.program_id(1)*TILE+tl.arange(0,TILE)
        if PACK:
            tl.store(y+placed*H+j,tl.load(x+(route//8)*H+j))
        else:
            tl.store(y+route*H+j,tl.load(x+placed*H+j))


@triton.jit
def _activation(x,y,ids,ranks,scales,H:tl.constexpr,TILE:tl.constexpr):
    route=tl.program_id(0)
    pos=tl.load(ranks+route)
    if pos>=0:
        j=tl.program_id(1)*TILE+tl.arange(0,TILE)
        e=tl.load(ids+route)
        scale=tl.load(scales+e).to(tl.float32)
        gate=(tl.load(x+pos*(2*H)+j).to(tl.float32)*scale).to(tl.bfloat16).to(tl.float32)
        up=(tl.load(x+pos*(2*H)+H+j).to(tl.float32)*scale).to(tl.bfloat16).to(tl.float32)
        exp=tl.inline_asm_elementwise("ex2.approx.ftz.f32 $0, $1;",constraints="=f,f",args=[-gate*1.4426950408889634],dtype=tl.float32,is_pure=True,pack=1)
        value=tl.inline_asm_elementwise("div.approx.ftz.f32 $0, $1, $2;",constraints="=f,f,f",args=[gate,1.+exp],dtype=tl.float32,is_pure=True,pack=1)
        tl.store(y+pos*H+j,value*up)


def mapping(ids,cutoff):
    m=ids.size(0)
    device=ids.device
    slots=m*8
    light=min(slots,32*max(cutoff,0))
    chunks=triton.cdiv(slots,256)
    ranks=torch.empty(slots,device=device,dtype=torch.int32)
    partial=torch.empty((chunks,32),device=device,dtype=torch.int32)
    offsets=torch.empty(66,device=device,dtype=torch.int32)
    base=torch.empty(32,device=device,dtype=torch.int32)
    _local_ids[32,chunks](ids,partial,ranks,slots,256)
    prefix=torch.cumsum(partial,dim=0,dtype=torch.int32) if chunks>1 else partial
    _offsets[(1,)](prefix,offsets,base,chunks,light,slots,cutoff)
    _complete[(chunks,)](ids,ranks,prefix,base,slots,256)
    return ranks,offsets,light


def compute(layer,x,ids,split,variant,scores,norm):
    m,h=x.shape
    slots=m*8
    assert ids.shape==(m,8) and h==4096
    plan=split._choose_large_prefill_configs(layer,slots,256)
    if plan is None:
        fixed={s:next(c for lo,hi,c in table if lo<slots<=hi) for s,table in layer.fp32_configs.items()}
        plan=dict(light=fixed,heavy=fixed,cutoff=-1)
    cutoff=plan["cutoff"] if variant=="split_fused" else -1
    ranks,offsets,light=mapping(ids,cutoff)
    rows=light+slots
    concat=x.new_empty((rows,2*h))
    up=concat
    down=concat.view(-1)[:rows*h].view(rows,h)
    inputs=x.new_empty((rows,h))
    # The gathered input can be reused for activation after all W13 reads complete.
    _copy[(slots,4)](x,inputs,ranks,h,True,1024)
    for stage,inp,out in (("w13",inputs,up),("w2",inputs,down)):
        if stage=="w2":
            _activation[(slots,4)](up,inputs,ids,ranks,layer.w13_post_scale,h,1024)
        parts=[]
        if light:
            parts.append((0,light,offsets[:33],plan["light"][stage]))
        cfg=plan["light" if variant=="single_light" else "heavy"][stage]
        parts.append((light,rows,offsets[33:],cfg))
        for lo,hi,layout,cfg in parts:
            HummingMethod.forward_layer(layer,inputs=inp[lo:hi],outputs=out[lo:hi],top_k=1,
                sublayer_name=stage,tuning_config=cfg,expert_layout=layout,valid_shape_m=hi-lo,
                compute_config={"gemm_type":"grouped_contiguous","use_f16_accum":False})
    normalized=x.new_empty((m,h))
    _mapped_finalize[(m,)](down,norm,scores,normalized,ids,ranks,layer.w2_post_scale,
        HAS_SCALE=True,TOPK=8,H=h,eps=1e-8,APPLY_NORM=True,HAS_WEIGHT=True,WEIGHT_FIRST=True,
        HAS_IDS=True,ROUND_PRODUCT=False,BLOCK_H=h,num_warps=8,num_stages=2)
    return normalized
@triton.jit
def _mapped_finalize(
    x_ptr,
    w_ptr,
    s_ptr,
    out_ptr,
    ids_ptr,
    ranks_ptr,
    scale_ptr,
    HAS_SCALE: tl.constexpr,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    eps,
    APPLY_NORM: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    WEIGHT_FIRST: tl.constexpr,
    HAS_IDS: tl.constexpr,
    ROUND_PRODUCT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    tok = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    if HAS_WEIGHT:
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for k in tl.static_range(TOPK):
        base = tl.load(ranks_ptr + tok * TOPK + k) * H
        live = True
        expert_id = -1
        if HAS_IDS:
            expert_id = tl.load(ids_ptr + tok * TOPK + k)
            live = expert_id >= 0
        if live:
            x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            if HAS_SCALE:
                scale = tl.load(scale_ptr + expert_id).to(tl.float32)
                # Preserve BF16 multiplication before FP32 norm/reduction.
                x = (x * scale).to(tl.bfloat16).to(tl.float32)

            if APPLY_NORM:
                if HAS_WEIGHT and WEIGHT_FIRST:
                    y = x * w
                    var = tl.sum(y * y, axis=0) / H
                    y = y * tl.rsqrt(var + eps)
                elif HAS_WEIGHT:
                    var = tl.sum(x * x, axis=0) / H
                    y = x * tl.rsqrt(var + eps) * w
                else:
                    var = tl.sum(x * x, axis=0) / H
                    y = x * tl.rsqrt(var + eps)
            else:
                y = x

            s = tl.load(s_ptr + tok * TOPK + k).to(tl.float32)
            product = y * s
            if ROUND_PRODUCT:
                product = product.to(x_ptr.dtype.element_ty).to(tl.float32)
            acc += product

    tl.store(out_ptr + tok * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


