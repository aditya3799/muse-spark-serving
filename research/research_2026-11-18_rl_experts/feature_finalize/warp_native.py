"""Compare native W2 feature-pack arithmetic with full original expert placements."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1]
sys.path.insert(0,str(root/"research_2026-11-18_rl/feature_shard"))
import native_gpu as src
from native_gpu import torch,finish
from warp_band import indices,bands_from_full,restore,candidate
from finalize import joined,paired_scalar_emulation
from native_certificate import certificate
import triton
import triton.language as tl


@triton.jit
def _full_sum(D,IDS,NORM,SCALE,OUT,H:tl.constexpr):
    slot=tl.program_id(0);e=tl.load(IDS+slot)
    v=0.
    if e>=0:
        f=tl.arange(0,H)
        x=tl.load(D+slot*H+f).to(tl.float32)
        x=(x*tl.load(SCALE+e).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        y=x*tl.load(NORM+f).to(tl.float32)
        v=tl.sum(y*y,0)
    tl.store(OUT+slot,v)

p=argparse.ArgumentParser()
p.add_argument("--out",type=Path,required=True);p.add_argument("--seed",type=int,required=True)
p.add_argument("--rows",type=int,nargs="+",default=[8,128]);p.add_argument("--experts",type=int,default=16)
p.add_argument("--wide-stress",action="store_true")
args=p.parse_args()
h=4096;e=args.experts
torch.set_num_threads(2);torch.manual_seed(args.seed)
norm=torch.linspace(.7,1.3,h,device="cuda",dtype=torch.bfloat16)
report=dict(seed=args.seed,expert_count=e,hidden=h,baseline_sha256=src.BASE,complete=False,cases=[],tiers={},
    provenance="Native W13 and W2 at full K=4096; serial two-rank emulation on RTX3050. The collected counters separate raw parent-vs-shard GEMM, restoring the original owner contributions and same-raw norm errors. No link timing.")
def save():
    tmp=args.out.with_suffix(".temp");tmp.write_text(json.dumps(report,indent=2));tmp.replace(args.out)
def cmp(a,b):
    return dict(mismatch=int(torch.count_nonzero(a.view(torch.int16)!=b.view(torch.int16))),
        n=a.numel(),finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        rel_l2=float((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-15)))
def original(raw,ids,scores,scales):
    m=ids.shape[0];parts=[]
    for r in (0,1):
        owned=(ids>=r*(e//2))&(ids<(r+1)*(e//2))
        parts.append(finish(raw.view(m,8,h),scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=torch.where(owned,ids,-1),expert_output_scale=scales))
    return parts[0]+parts[1],torch.stack(parts)
def reference(shards,ids,scores,unused_scale):
    return original(torch.cat(shards,-1),ids,scores,unused_scale)
def measure(raw,ids,scores,scale,band_parts=None):
    m=ids.shape[0]
    band_parts=bands_from_full(raw) if band_parts is None else band_parts
    band_norm=bands_from_full(norm)
    output,owners,total=candidate(band_parts,ids,scores,band_norm,scale)
    baseline,ownerref=original(raw,ids,scores,scale)
    contiguous=[v.contiguous() for v in raw.chunk(2,-1)]
    direct,direct_owners=joined(contiguous,ids,scores,norm,scale)
    scalar,scalar_owners=paired_scalar_emulation(contiguous,ids,scores,norm,scale)
    full=torch.empty(8*m,device="cuda")
    _full_sum[(8*m,)](raw,ids,norm,scale,full,h,num_warps=8,num_stages=2)
    r=dict(band_final=cmp(baseline,output),band_owners=cmp(ownerref,owners),
        direct_final=cmp(baseline,direct),direct_owners=cmp(ownerref,direct_owners),
        contiguous_scalar_final=cmp(baseline,scalar),contiguous_scalar_owners=cmp(ownerref,scalar_owners),
        band_square_sum_f32_mismatches=int((total.view(torch.int32)!=full.view(torch.int32)).sum()))
    return r,total,full
def main():
    weights=src.make_weights(e,h)
    order=torch.cat(indices(h))
    w,s,glob=weights["w2"]
    alt={**weights,"w2":(w.index_select(1,order),s.index_select(1,order),glob)}
    with src.cached_threshold_for_preparation(here):
        parents=[src.pack(weights,i,False) for i in (0,1)]
        layers=[src.pack(alt,i,True) for i in (0,1)]
    del weights,alt,w,s,glob
    scales=layers[0].w2_post_scale
    for m in args.rows:
        for dist in ("normal","skewed"):
            for i in range(2):
                torch.manual_seed(args.seed+37*m+3*i+(int(dist=="skewed")*1009))
                ids=torch.randint(e*4,(m,8),device="cuda",dtype=torch.int32)
                if dist=="skewed":ids[:,0]=0;ids[::2,1]=e//2
                ids=torch.where(ids<e,ids,-1)
                x=torch.randn((m,h),device="cuda",dtype=torch.bfloat16)*.02
                scores=torch.rand((m,8),device="cuda");scores/=scores.sum(-1,keepdim=True)
                hidden=torch.cat([src.stage(layer,"w13",x,ids) for layer in layers],dim=-1)
                band_parts=[src.stage(layer,"w2",hidden,ids) for layer in layers]
                raw=restore(band_parts)
                result,_,_=measure(raw,ids,scores,scales,band_parts)
                fullref,shardowners=original(raw,ids,scores,scales)
                orig_owners=[];raw_counts=[]
                for rank,layer in enumerate(parents):
                    loc=ids-rank*(e//2);loc=torch.where((loc>=0)&(loc<(e//2)),loc,-1)
                    down,_=src.native["_humming_expert_outputs"](layer,x,loc,torch,global_experts=4*e,defer_downscale=True)
                    live=loc>=0
                    raw_counts.append(cmp(down[live],raw.view(m,8,h)[live]))
                    orig_owners.append(finish(down,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=loc,expert_output_scale=layer.w2_post_scale))
                boutput,bowners,_=candidate(band_parts,ids,scores,bands_from_full(norm),scales)
                result.update(rows=m,dist=dist,i=i,raw_parent_vs_feature=raw_counts,
                    end_to_end_owner=cmp(torch.stack(orig_owners),bowners),end_to_end_output=cmp(orig_owners[0]+orig_owners[1],boutput))
                if i==0:result["contiguous_certificate"]=certificate([v.contiguous() for v in raw.chunk(2,-1)],ids,scores,norm,scales,reference)
                report["cases"].append(result);save()
                print(m,dist,i,"same_raw",result["band_final"]["mismatch"],"sum",result["band_square_sum_f32_mismatches"],"e2e",result["end_to_end_output"]["mismatch"],flush=True)
    m=32;ids=torch.full((m,8),-1,device="cuda",dtype=torch.int32);ids[:,0]=torch.arange(m,device="cuda")%e
    raw=torch.zeros((m,8,h),device="cuda",dtype=torch.bfloat16)
    raw[::2,0,0]=4096.;raw[1::2,0,8]=4096.;raw[::2,0,8]=1.;raw[1::2,0,0]=1.;raw[:,0,h//2]=1.
    saved=norm.clone();norm.fill_(1.)
    scores=torch.zeros((m,8),device="cuda");scores[:,0]=.505859375;scores[:,1]=1.-scores[:,0]
    r,band,full=measure(raw.reshape(-1,h),ids,scores,torch.ones_like(scales))
    report["explicit"]=dict(result=r,full_sum=full.view(m,8)[:2,0].tolist(),band_sum=band.view(m,8)[:2,0].tolist());norm.copy_(saved)
    if args.wide_stress:
        torch.manual_seed(args.seed+321)
        rs=[]
        for j in range(4):
            v=torch.randn((m*8,h),device="cuda",dtype=torch.bfloat16)
            v[:,1::8]*=2.**10;v[:,3::8]*=2.**-12
            raw=v.reshape(m,8,h)
            ids=torch.randint(-1,e,(m,8),device="cuda",dtype=torch.int32);scores.uniform_();scores/=scores.sum(-1,keepdim=True)
            rr,_,_=measure(raw.reshape(-1,h),ids,scores,scales);rs.append(rr)
        report["stress"]=rs
    report["complete"]=True;save();print("explicit",report["explicit"],flush=True)
if __name__=="__main__":
    with torch.inference_mode():main()
