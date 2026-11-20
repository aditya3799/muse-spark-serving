"""Report conservative repair granularity on real W2 outputs outside timing."""
import torch
from finalize import local_squares,joined,paired_scalar_emulation


def certificate(shards,ids,scores,norm,scale,reference):
    m=ids.shape[0];h=norm.numel();e=scale.numel()
    halves=norm.chunk(2)
    sums=[local_squares(d,ids,n,scale) for d,n in zip(shards,halves)]
    total=(sums[0]+sums[1]).reshape(m,8)
    inv=torch.rsqrt(total/h+1.e-8)
    acc=torch.zeros((2,m,h),device=ids.device)
    mass=torch.zeros_like(acc)
    okay=torch.ones(m,dtype=torch.bool,device=ids.device)
    whole=torch.cat(shards,-1).view(m,8,h)
    valid=ids>=0
    for j in range(8):
        raw=whole[:,j]
        post=scale[ids[:,j].clamp_min(0).long()]
        y=(raw.float()*post[:,None].float()).bfloat16().float()*norm.float()
        live=valid[:,j]
        finite=(y==0)|((y.abs()>=2.**-50)&(y.abs()<=2.**50))
        s=scores[:,j]
        okay &= (~live)|(finite.all(1)&((s==0)|((s.abs()>=2.**-20)&(s.abs()<=1))))
        projected=y*inv[:,j,None]
        v=projected*s[:,None]
        for g in (0,1):
            use=live&((ids[:,j]//(e//2))==g)
            # Match the compiler's fused accumulator instead of inferring it
            # from a cast BF16 owner result.
            combined=torch.addcmul(acc[g],projected,s[:,None])
            acc[g]=torch.where(use[:,None],combined,acc[g])
            mass[g]+=torch.where(use[:,None],v.abs(),0.)
    result={}
    actual,parts=paired_scalar_emulation(shards,ids,scores,norm,scale)
    _,parent_parts=reference(shards,ids,scores,scale)
    differ=parts.view(torch.int16)!=parent_parts.view(torch.int16)
    acc_center_matches=torch.equal(acc.bfloat16().view(torch.int16),parts.view(torch.int16))
    for lab,q in (("conservative",2.**-17),("optimistic_unproved",2.**-20)):
        radius=q*mass+32*torch.finfo(torch.float32).tiny
        lo=torch.nextafter(acc-radius,torch.full_like(acc,-float("inf"))).bfloat16()
        hi=torch.nextafter(acc+radius,torch.full_like(acc,float("inf"))).bfloat16()
        amb=((lo.view(torch.int16)!=hi.view(torch.int16))&(mass!=0)) | (~okay)[None,:,None]
        own=amb.any(2)
        routes=sum(int((valid&(ids//(e//2)==g)&own[g,:,None]).sum()) for g in (0,1))
        result[lab]=dict(active_slots=int(valid.sum()),ambiguous_original_owner_routes=routes,
            ambiguous_features=int(amb.sum()),observed_differences_outside_mask=int((differ&~amb).sum()),
            estimated_center_bf16_matches_scalar_owner=acc_center_matches)
    return result
