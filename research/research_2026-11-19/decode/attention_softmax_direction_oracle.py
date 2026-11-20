"""CPU mechanism study: pinned FA3 reverse traversal vs donor forward traversal.
Not GPU execution, WGMMA ordering, exp2 approximation, or serving validation.
Preserves prior same-direction oracle as a tile-size-only ablation.
"""
import hashlib,json,math
from pathlib import Path
import torch
torch.set_num_threads(2)
HERE=Path(__file__).resolve().parent
SEED=8371931;torch.manual_seed(SEED)
def online(scores,values,tile,reverse):
 rows,n=scores.shape;maximum=torch.full((rows,),-torch.inf)
 sums=torch.zeros(rows);out=torch.zeros(rows,values.shape[1])
 starts=list(range(0,n,tile));starts=starts[::-1] if reverse else starts
 for ix,begin in enumerate(starts):
  end=min(n,begin+tile);s=scores[:,begin:end];v=values[begin:end]
  current=torch.maximum(maximum,s.amax(1));safe=torch.where(torch.isneginf(current),0.,current)
  alpha=torch.ones_like(safe) if ix==0 else torch.exp2(maximum-safe)
  out*=alpha[:,None];sums*=alpha
  shift=torch.where(torch.isneginf(current),0.,current-8.)
  p=torch.exp2(s-shift[:,None]);out+=p.to(torch.float8_e4m3fn).float()@v
  sums+=p.sum(1);maximum=current
 return out/sums.clamp_min(torch.finfo(torch.float32).tiny)[:,None]
def stats(out,ref):
 delta=out-ref
 return dict(relative_l2=float(delta.norm()/ref.norm()),max_abs=float(delta.abs().max()),
  bf16_changed=int((out.bfloat16()!=ref.bfloat16()).sum()),elements=out.numel(),finite=bool(out.isfinite().all()))
records=[]
for n in (128,2048,8192):
 q=torch.randn(16,64).bfloat16();k=torch.randn(n,64).bfloat16()
 q=torch.nn.functional.rms_norm(q,(64,),eps=1e-5).to(torch.float8_e4m3fn).float()
 k=torch.nn.functional.rms_norm(k,(64,),eps=1e-5).to(torch.float8_e4m3fn).float()
 v=torch.randn(n,64).bfloat16().to(torch.float8_e4m3fn).float()
 scores=(q@k.T)*(math.log2(math.e)/8*5.4730064863875)
 valid=torch.linspace(0,n,16).long();scores[torch.arange(n)[None,:]>=valid[:,None]]=-torch.inf
 exact=torch.softmax(scores.double()*math.log(2),1).nan_to_num()@v.double();exact=exact.float()
 native=online(scores,v,128,True);same_direction=online(scores,v,64,True);candidate=online(scores,v,64,False)
 records.append(dict(n=n,native_reverse_vs_fp64=stats(native,exact),
  restored_forward_vs_fp64=stats(candidate,exact),restored_forward_vs_native_reverse=stats(candidate,native),
  tile64_reverse_vs_native_reverse=stats(same_direction,native)))
doc=dict(scope=__doc__,seed=SEED,softmax_scale=5.4730064863875/8,
 native_direction='reverse BlockN128 per frozenFA3mainloop',candidate_direction='forward BlockN64 per donor start+i*TILE loop',
 rows=records,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),serving_qualified=False)
(HERE/'attention_softmax_direction_oracle.json').write_text(json.dumps(doc,indent=2))
for r in records:print(r['n'],{k:round(v['relative_l2'],6) for k,v in r.items() if isinstance(v,dict)})
