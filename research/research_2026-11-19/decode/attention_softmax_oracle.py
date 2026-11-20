"""CPU arithmetic study of FP8 probability scaling and tiled online softmax.

This isolates the probability/normalization mechanism. It cannot reproduce
WGMMA/FA3 reduction order, approximate GPU exponentials, or claim kernel parity.
"""
import hashlib,json,math
from pathlib import Path
import torch
torch.set_num_threads(2)
HERE=Path(__file__).resolve().parent
SEED=9185713;torch.manual_seed(SEED)

def online(scores,values,tile,mode):
 rows,n=scores.shape;maximum=torch.full((rows,),-torch.inf)
 sums=torch.zeros(rows);out=torch.zeros(rows,values.shape[1]);changes=0
 # FA3 traverses K from the final tile backwards.
 for begin in reversed(range(0,n,tile)):
  end=min(n,begin+tile);s=scores[:,begin:end];v=values[begin:end]
  m=s.amax(1);first=end==n
  if mode=='donor_lazy':
   update=first | (m>maximum+8.)
   current=torch.where(update,torch.maximum(m,maximum),maximum)
   offset=0.
  else:current=torch.maximum(maximum,m);offset=8.
  safe=torch.where(torch.isneginf(current),0.,current)
  alpha=torch.ones_like(safe) if first else torch.exp2(maximum-safe)
  out*=alpha[:,None];sums*=alpha
  shift=torch.where(torch.isneginf(current),0.,current-offset)
  p=torch.exp2(s-shift[:,None]);q=p.to(torch.float8_e4m3fn).float()
  sums+= (q if mode=='donor_lazy' else p).sum(1)
  out+=q@v;maximum=current
 return out/sums.clamp_min(torch.finfo(torch.float32).tiny)[:,None]

def stats(out,reference):
 error=out-reference
 return dict(max_abs=float(error.abs().max()),rms=float(error.square().mean().sqrt()),
  relative_l2=float(error.norm()/reference.norm()),bf16_changed=int((out.bfloat16()!=reference.bfloat16()).sum()),elements=out.numel(),finite=bool(out.isfinite().all()))

records=[]
for n in (128,2048,8192):
 for scale in (1.,4.):
  q=torch.randn(16,64).to(torch.float8_e4m3fn).float()
  k=torch.randn(n,64).to(torch.float8_e4m3fn).float()
  v=torch.randn(n,64).to(torch.float8_e4m3fn).float()
  scores=(q@k.T)*(math.log2(math.e)/8*scale)
  # Ragged causal rows, including one fully masked row.
  valid=torch.linspace(0,n,16).long();scores[torch.arange(n)[None,:]>=valid[:,None]]=-torch.inf
  exact=torch.softmax(scores.double()*math.log(2),1).nan_to_num()@v.double();exact=exact.float()
  # Pinned heuristics.h selects one MMA WG: q_len8 * GQA8 <=64;
  # tile_size.h chooses FP8 BlockN128, not its multi-WG default160.
  native=online(scores,v,128,'native')
  candidate=online(scores,v,64,'native')
  donor=online(scores,v,64,'donor_lazy')
  records.append(dict(n=n,logit_multiplier=scale,native_fp8_vs_fp64=stats(native,exact),
   restored_fp8_vs_fp64=stats(candidate,exact),restored_vs_native=stats(candidate,native),
   donor_lazy_vs_fp64=stats(donor,exact)))
doc=dict(seed=SEED,scope=__doc__,rows=records,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(HERE/'attention_softmax_oracle.json').write_text(json.dumps(doc,indent=2))
for r in records:print(r['n'],r['logit_multiplier'],{k:round(r[k]['relative_l2'],6) for k in r if isinstance(r[k],dict)})
