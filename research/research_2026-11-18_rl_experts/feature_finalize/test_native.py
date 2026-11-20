"""Isolate finalized results on the identical native W2 shards and adversarial cutoffs."""
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
here=Path(__file__).resolve().parent
root=here.parents[1]
sys.path.insert(0,str(root/"research_2026-11-18_rl/feature_shard"))
import native_gpu as source
from native_gpu import torch,finish
import triton
import triton.language as tl
from finalize import joined,local_squares,paired_scalar_emulation
from native_certificate import certificate


@triton.jit
def _whole(D,NORM,SC,IDS,OUT,H:tl.constexpr):
    slot=tl.program_id(0);e=tl.load(IDS+slot)
    v=0.
    if e>=0:
        f=tl.arange(0,H)
        x=tl.load(D+slot*H+f).to(tl.float32)
        x=(x*tl.load(SC+e).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        y=x*tl.load(NORM+f).to(tl.float32)
        v=tl.sum(y*y,0)
    tl.store(OUT+slot,v)


p=argparse.ArgumentParser()
p.add_argument("--out",type=Path,required=True);p.add_argument("--seed",type=int,required=True)
p.add_argument("--rows",type=int,nargs="+",default=[32,128]);p.add_argument("--width",type=int,default=4096)
p.add_argument("--experts",type=int,default=16);p.add_argument("--rounds",type=int,default=8)
args=p.parse_args()
torch.set_num_threads(2);torch.manual_seed(args.seed)
h=args.width;e=args.experts
norm=torch.linspace(.7,1.3,h,dtype=torch.bfloat16,device="cuda")
report=dict(seed=args.seed,expected_scope="Same raw native W2 feature outputs supplied to every finalizer. Local graph synthetic collective sums; no communication timing and no cross-GEMM parity claim.",
    native_w2_fixture_sha256=hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(), finalized_source_sha256=hashlib.sha256((here/"finalize.py").read_bytes()).hexdigest(),raw_native=[],stress=[],paired={},complete=False)
def save():
    tmp=args.out.with_suffix(".temp");tmp.write_text(json.dumps(report,indent=2));tmp.replace(args.out)
def compare(a,b):
    err=(a.view(torch.int16)!=b.view(torch.int16))
    return dict(bits=int(err.sum()),finite=bool(torch.isfinite(b).all()),max_abs=float((a.float()-b.float()).abs().max()))
def reference(shards,ids,scores,scale):
    m=ids.shape[0];down=torch.cat(shards,-1).view(m,8,h);owner=[]
    for rank in (0,1):
        mask=(ids>=rank*(e//2))&(ids<(rank+1)*(e//2))
        owner.append(finish(down,scores,norm_weight=norm,eps=1e-8,weight_first=True,
            slot_expert_ids=torch.where(mask,ids,-1),expert_output_scale=scale))
    return owner[0]+owner[1],torch.stack(owner)
def measure_case(m,shards,ids,scores,scale,label):
    answers=[f(shards,ids,scores,norm,scale) for f in (joined,paired_scalar_emulation)]
    ref,owner=reference(shards,ids,scores,scale)
    rec={"rows":m,"label":label}
    for key,(out,parts) in zip(("joined","scalar"),answers):
        rec[key]={"final":compare(ref,out),"owners":compare(owner,parts.contiguous())}
    return rec
def adversarial(scale):
    # Complete-FFN ordinary scores rarely expose a one-ULP difference. Restrict
    # a token to one live route and choose a legal positive gate near a BF16 midpoint.
    m=96;ids=torch.full((m,8),-1,device="cuda",dtype=torch.int32);ids[:,0]=torch.arange(m,device="cuda")%e
    scores=torch.zeros((m,8),device="cuda")
    assert h==4096
    raw=torch.zeros((m,8,h),device="cuda",dtype=torch.bfloat16)
    raw[::2,0,0]=4096.;raw[1::2,0,8]=4096.
    raw[::2,0,8]=1.;raw[1::2,0,0]=1.
    raw[:,0,h//2]=1.
    parts=[r.contiguous() for r in raw.view(-1,h).chunk(2,-1)]
    one=torch.ones_like(norm);sc=torch.ones_like(scale)
    scores[:,0]=.505859375;scores[:,1]=1.-scores[:,0]
    saved=norm.clone();norm.copy_(one)
    report["explicit"]=measure_case(m,parts,ids,scores,sc,"finite norm pair with identical scalar-partials")
    full=torch.empty(m*8,device="cuda");_whole[(m*8,)](raw,one,sc,ids,full,h,num_warps=8,num_stages=2)
    halfsum=local_squares(parts[0],ids,one[:h//2],sc)+local_squares(parts[1],ids,one[h//2:],sc)
    report["explicit"]["full_sums"]=full.view(m,8)[:2,0].tolist()
    report["explicit"]["scalar_sums"]=halfsum.view(m,8)[:2,0].tolist()
    norm.copy_(saved);save()
    for seed in range(3):
        torch.manual_seed(args.seed+100*seed)
        raw=torch.randn((m*8,h),device="cuda",dtype=torch.bfloat16)
        raw[:,1::4]*=.03125;raw[:,3::8]*=8.
        shards=[v.contiguous() for v in raw.chunk(2,-1)]
        sq=torch.empty(m*8,device="cuda",dtype=torch.float32)
        _whole[(m*8,)](raw,norm,scale,ids,sq,h,num_warps=8,num_stages=2)
        a=local_squares(shards[0],ids,norm[:h//2],scale);b=local_squares(shards[1],ids,norm[h//2:],scale)
        full=sq.view(m,8)[:,0];split=(a+b).view(m,8)[:,0]
        for col in (0,6,15):
            v=(raw.view(m,8,h)[:,0,col].float()*scale[ids[:,0].long()].float()).bfloat16().float()*norm[col].float()
            y=v*torch.rsqrt(full/h+1.e-8)
            pivot=(y.abs()*.375).bfloat16()
            next_value=(pivot.view(torch.int16)+1).view(torch.bfloat16)
            midpoint=(pivot.float()+next_value.float())*.5
            base=(midpoint/y.abs().clamp(min=1.e-24)).clamp(max=.9)
            for step in (-1,0,1):
                trial=(base.view(torch.int32)+step).view(torch.float32).clamp(min=0,max=.99)
                scores[:,0]=trial;scores[:,1]=1.-trial
                r=measure_case(m,shards,ids,scores,scale,f"seed{seed}_col{col}_ulp{step}")
                r["unequal_f32_sums"]=int((full.view(torch.int32)!=split.view(torch.int32)).sum())
                report["stress"].append(r);save()
                if r["scalar"]["final"]["bits"] and report.get("counterexample") is None:
                    baseline,_=reference(shards,ids,scores,scale);test,_=paired_scalar_emulation(shards,ids,scores,norm,scale)
                    ij=torch.nonzero(baseline.view(torch.int16)!=test.view(torch.int16))[0].tolist()
                    row,c=ij
                    report["counterexample"]=dict(seed=args.seed+100*seed,row=row,col=c,chosen_col=col,step=step,
                        score=float(scores[row,0]),flat_reference_output=float(baseline[row,c]),scalar_output=float(test[row,c]),original_square_sum=float(full[row]),sharded_square_sum=float(split[row]))
                if r["joined"]["final"]["bits"]:return
        if report.get("counterexample"):break
def main():
    weights=source.make_weights(e,h)
    with source.cached_threshold_for_preparation(here):
        layers=[source.pack(weights,i,True) for i in (0,1)]
    del weights
    scale=layers[0].w2_post_scale
    for m in args.rows:
        for pattern in ("uniform","skewed"):
            torch.manual_seed(args.seed+31*m+(1 if pattern=="skewed" else 0))
            ids=torch.randint(e*4,(m,8),device="cuda",dtype=torch.int32)
            if pattern=="skewed":ids[:,0]=0;ids[::2,1]=e//2
            ids=torch.where(ids<e,ids,-1)
            scores=torch.rand((m,8),device="cuda");scores/=scores.sum(1,keepdim=True)
            x=torch.randn((m,h),device="cuda",dtype=torch.bfloat16)*.02
            hidden=torch.cat([source.stage(layer,"w13",x,ids) for layer in layers],-1)
            parts=[source.stage(layer,"w2",hidden,ids) for layer in layers]
            native=measure_case(m,parts,ids,scores,scale,pattern)
            native["interval"]=certificate(parts,ids,scores,norm,scale,reference)
            report["raw_native"].append(native);save()
            funcs=[lambda:reference(parts,ids,scores,scale),lambda:joined(parts,ids,scores,norm,scale),lambda:paired_scalar_emulation(parts,ids,scores,norm,scale)]
            graphs=[];outputs=[]
            for fn in funcs:
                fn();torch.cuda.synchronize()
                with torch.cuda.graph(g:=torch.cuda.CUDAGraph()):outputs.append(fn())
                graphs.append(g);g.replay()
            record={k:[] for k in ("baseline","joined","scalar","control")}
            for j in range(args.rounds):
                seq=["baseline","joined","scalar","control"] if j%2==0 else ["control","scalar","joined","baseline"]
                for tag in seq:
                    i={"baseline":0,"joined":1,"scalar":2,"control":0}[tag]
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record();graphs[i].replay();b.record();b.synchronize();record[tag].append(a.elapsed_time(b)*1000)
            report["paired"][f"M{m}_{pattern}"]=record
            save();print(m,pattern,report["raw_native"][-1],flush=True)
    adversarial(scale);report["complete"]=True;save()
if __name__=="__main__":
    with torch.inference_mode():main()
