"""CPU feasibility of a cancellation-aware output interval, isolated from GPU timing."""
import json
from pathlib import Path
import numpy as np
here=Path(__file__).resolve().parent
def bf16(a):
    a=np.asarray(a,dtype=np.float32)
    u=a.view(np.uint32)
    return ((u+32767+((u>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
def run(seed,mode):
    rng=np.random.default_rng(seed)
    m,h,e=128,4096,16
    ids=rng.integers(0,4*e,size=(m,8))
    if mode=="skewed":
        ids[:,0]=0;ids[::2,1]=e//2
    ids=np.where(ids<e,ids,-1)
    sc=rng.random((m,8),dtype=np.float32);sc/=sc.sum(-1,keepdims=True)
    scale=bf16(np.linspace(.125,.5,e,dtype=np.float32))
    norm=bf16(np.linspace(.7,1.3,h,dtype=np.float32))
    acc=np.zeros((2,m,h),dtype=np.float32)
    mass=np.zeros_like(acc)
    eligible=np.ones(m,dtype=bool)
    for j in range(8):
        live=ids[:,j]>=0
        raw=bf16(rng.normal(size=(m,h)).astype(np.float32))
        y=bf16(raw*scale[ids[:,j],None])*norm
        eligible &= np.all((y==0)|((np.abs(y)>=2.**-50)&(np.abs(y)<=2.**50)),axis=1)
        eligible &= (sc[:,j]==0)|((np.abs(sc[:,j])>=2.**-20)&(np.abs(sc[:,j])<=1))
        squares=y*y
        halves=[q.sum(1,dtype=np.float32) for q in np.split(squares,2,axis=1)]
        inv=1./np.sqrt((halves[0]+halves[1])/h+1.e-8)
        v=(y*inv[:,None])*sc[:,j,None]
        for g in (0,1):
            select=live & (ids[:,j]//(e//2)==g)
            acc[g,select]+=v[select]
            mass[g,select]+=np.abs(v[select])
    active=(ids>=0).any(1)
    tally={}
    for name,rel in (("conservative",2.**-17),("optimistic_unproved",2.**-20)):
        # Outward rounding after calculating a cancellation-aware absolute radius.
        rad=rel*mass+32*np.finfo(np.float32).tiny
        low=np.nextafter(acc-rad,-np.inf,dtype=np.float32)
        high=np.nextafter(acc+rad,np.inf,dtype=np.float32)
        uncertain=(bf16(low).view(np.uint32)!=bf16(high).view(np.uint32))&(mass!=0)
        owner=uncertain.any(2)|(~eligible)[None,:]
        token=owner.any(0)
        owner_bytes=sum((((ids>=0)&(ids//(e//2)==g)) & owner[g,:,None]).sum() for g in (0,1))
        # A repair needs the complete rank slices of all active routes of a
        # flagged token when the original EP-owner contributions might differ.
        tally[name]=dict(absolute_radius_rel_to_mass=rel,flagged_features=int(uncertain.sum()),
            active_tokens=int(active.sum()),ambiguous_active_tokens=int((token&active).sum()),
            ambiguous_tokens_fraction=float((token&active).sum()/active.sum()),
            fraction_live_route_bytes_with_owner_repair=float(owner_bytes/(ids>=0).sum()),
            fraction_live_route_bytes_to_repair=float(((ids>=0)&token[:,None]).sum()/(ids>=0).sum()))
    return dict(seed=seed,mode=mode,eligible_domain_tokens=int(eligible.sum()),total_tokens=m,results=tally)
results=[run(s,p) for s in (37199,68281) for p in ("uniform","skewed")]
doc=dict(scope="CPU feasibility on generated candidate W2 BF16 values, not native outputs, device timing, or H200 data. Certified zero-contribution underflow/special cases require a production contract or exact fallback.",rows=results,
    bound="Use FP32 unit roundoff u=2^-24, gamma16<9.54e-7 on nonnegative square sums. The observed full and two-half Triton pairwise trees each need at most 16 rounded operations along an input path, including its square and the pair add. H is a power of two. Adding epsilon is rounded once. PTX rsqrt.approx.f32 relative error <=2^-22.9. Each signed 8-slot owner accumulation contributes at most gamma10 times the sum of absolute terms. Twice these one-path errors plus endpoint/absolute-sum rounding fit under 2^-17 times mass, with 32 normal-minima added for tiny cancellation residuals. Unqualified inputs or another unverified compiler layout require exact fallback.",
    source="https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rsqrt")
(here/"certificate.json").write_text(json.dumps(doc,indent=2))
print([(r["mode"],r["results"]["conservative"]["fraction_live_route_bytes_to_repair"],r["results"]["optimistic_unproved"]["fraction_live_route_bytes_to_repair"]) for r in results])
