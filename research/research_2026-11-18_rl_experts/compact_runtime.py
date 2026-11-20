"""Fixed-capacity whole-token buckets for Humming W13/W2, no host route counts."""
import torch
import triton
import triton.language as tl

@triton.jit
def _pack(X,IDS,LIGHT,BEFORE,PX,PI,PL,PH,BUCKET:tl.constexpr,STRIDE:tl.constexpr,H:tl.constexpr,TILE:tl.constexpr):
    token=tl.program_id(0)
    first=tl.load(BEFORE+token)
    if first//STRIDE==BUCKET:
        slot=tl.program_id(1)
        lanes=tl.arange(0,8)
        rank_ids=tl.load(IDS+token*8+lanes)
        prior=tl.sum(((rank_ids>=0)&(lanes<slot)).to(tl.int32),axis=0)
        expert=tl.load(IDS+token*8+slot)
        if expert>=0:
            pos=first-BUCKET*STRIDE+prior
            feature=tl.program_id(2)*TILE+tl.arange(0,TILE)
            tl.store(PX+pos*H+feature,tl.load(X+token*H+feature))
            if tl.program_id(2)==0:
                lid=tl.load(LIGHT+token*8+slot)
                tl.store(PI+pos,expert)
                tl.store(PL+pos,lid)
                tl.store(PH+pos,tl.where(lid<0,expert,-1))

@triton.jit
def _finish(DOWN,IDS,BEFORE,NORM,SCORES,POST,OUT,G:tl.constexpr,STRIDE:tl.constexpr,H:tl.constexpr):
    token=tl.program_id(0)
    first=tl.load(BEFORE+token)
    if first//STRIDE==G:
        f=tl.arange(0,H)
        norm=tl.load(NORM+f).to(tl.float32)
        value=tl.full([H],0,tl.float32)
        packed=first-G*STRIDE
        for j in tl.static_range(8):
            expert=tl.load(IDS+token*8+j)
            if expert>=0:
                x=tl.load(DOWN+packed*H+f).to(tl.float32)
                # Same scored per-slot BF16 scale boundary and FP32 weighted RMS.
                w2scale=tl.load(POST+expert).to(tl.float32)
                x=(x*w2scale).to(tl.bfloat16).to(tl.float32)
                y=x*norm
                var=tl.sum(y*y,axis=0)/H
                y=y*tl.rsqrt(var+1.e-8)
                gate=tl.load(SCORES+token*8+j)
                value+=y*gate
                packed+=1
        tl.store(OUT+token*H+f,value)

def routed(layer,x,ids,scores,norm,split,scale_silu):
    from humming.layer import HummingMethod
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    m,h=x.shape
    assert ids.shape==(m,8) and h==4096 and m>=32
    valid=ids.numel()
    plan=split._choose_large_prefill_configs(layer,valid,256)
    assert plan is not None, "candidate currently qualified for split-plan source paths only"
    partition=split.partition_routes_scalable if valid>4096 else split.partition_routes
    light,heavy,counts=partition(ids,32,plan["cutoff"])
    n=(ids>=0).sum(dim=1,dtype=torch.int32)
    first=torch.cumsum(n,0,dtype=torch.int32)-n
    stride=2*m
    capacity=stride+7
    buckets=triton.cdiv(valid,stride)
    packed_x=torch.empty((capacity,h),device=x.device,dtype=x.dtype)
    large=torch.empty(capacity*h*2,device=x.device,dtype=x.dtype)
    up=large.view(capacity,2*h)
    gate=torch.empty((capacity,h),device=x.device,dtype=x.dtype)
    down=large[:capacity*h].view(capacity,h)
    pi,pl,ph=[torch.empty((capacity,1),device=x.device,dtype=torch.int32) for _ in range(3)]
    output=torch.empty_like(x)
    for b in range(buckets):
        # Clear all three sparse ID tensors, never any dense inactive rows.
        pi.fill_(-1);pl.fill_(-1);ph.fill_(-1)
        _pack[(m,8,4)](x,ids,light,first,packed_x,pi,pl,ph,b,stride,h,1024,num_warps=4)
        aligned={}
        for stage,inputs,outputs in (("w13",packed_x,up),("w2",gate,down)):
            if stage=="w2":
                scale_silu(up,gate,pi,layer.w13_post_scale)
            for kind,part in (("light",pl),("heavy",ph)):
                cfg=plan[kind][stage]
                block=cfg["block_shape"][0]
                if (kind,block) not in aligned:
                    aligned[kind,block]=moe_align_block_size(part,block,32,ignore_invalid_expert=True)
                sorted_ids,experts,padded=aligned[kind,block]
                HummingMethod.forward_layer(layer,inputs=inputs,outputs=outputs,top_k=1,sublayer_name=stage,
                    tuning_config=cfg,sorted_ids=sorted_ids,expert_ids=experts,num_tokens_padded=padded,
                    valid_shape_m=capacity,compute_config={"gemm_type":"indexed","use_f16_accum":False})
        _finish[(m,)](down,ids,first,norm,scores,layer.w2_post_scale,output,b,stride,h,num_warps=8,num_stages=2)
    return output

