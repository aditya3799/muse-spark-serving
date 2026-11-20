"""Independent correctness oracle for native SM86 SWA offset handling.

No performance or serving promotion. Keeps earlier negative receipts untouched.
"""
import hashlib
import inspect
import json
import math
from pathlib import Path
import torch
import sgl_kernel.flash_attn as fa

HERE=Path(__file__).resolve().parent
torch.manual_seed(3751381)
rows=[]
for heads in (2,16):
 for qlens in ((8,8),(1,8)):
  lengths=(2853,7171); cap=8192; d=64; kvheads=2
  k=torch.randn((16000,1,kvheads,d),device='cuda',dtype=torch.bfloat16)*.35
  v=torch.randn_like(k)*.35
  table=torch.stack([torch.randperm(7900,device='cuda')+100,torch.randperm(7900,device='cuda')+8000]).int()
  table=torch.nn.functional.pad(table,(0,cap-7900))
  lens=torch.tensor(lengths,device='cuda',dtype=torch.int32)
  cu=torch.tensor([0,qlens[0],sum(qlens)],device='cuda',dtype=torch.int32)
  for zero in (False,True):
   q=torch.randn((sum(qlens),heads,d),device='cuda',dtype=torch.bfloat16)*.31
   if zero:q.zero_()
   for left in (-1,2047):
    ref=[]; reflse=[]
    for i,(length,qn) in enumerate(zip(lengths,qlens)):
     query=q[int(cu[i]):int(cu[i+1])].float().transpose(0,1)
     keys=k[table[i,:length].long(),0].float().repeat_interleave(heads//kvheads,dim=1).permute(1,2,0)
     vals=v[table[i,:length].long(),0].float().repeat_interleave(heads//kvheads,dim=1).transpose(0,1)
     score=query@keys/math.sqrt(d)
     kp=torch.arange(length,device='cuda')[None,:]
     qp=torch.arange(qn,device='cuda')[:,None]+length-qn
     mask=kp<=qp
     if left>=0:mask &= kp>=qp-left
     score.masked_fill_(~mask[None,:,:],-float('inf'))
     ref.append((score.softmax(-1)@vals).transpose(0,1))
     reflse.append(score.logsumexp(-1).transpose(0,1))
    ref=torch.cat(ref);reflse=torch.cat(reflse)
    for packed in (False,True):
     for rebased in ((False,True) if left>=0 else (False,)):
      base=(lens-torch.tensor(qlens,device='cuda')-left).clamp_min(0).int() if rebased else torch.zeros_like(lens)
      ln=lens-base
      width=int(ln.max())
      cols=base[:,None]+torch.arange(width,device='cuda')[None,:]
      pt=table.gather(1,cols.long().clamp_max(cap-1)).contiguous() if rebased else table
      output,lse,*_=fa.flash_attn_with_kvcache(q,k,v,cache_seqlens=ln,page_table=pt,cu_seqlens_q=cu,
       max_seqlen_q=max(qlens),causal=True,window_size=(left,0),num_splits=1,pack_gqa=packed,
       return_softmax_lse=True,ver=3)
      # Native varlen LSE is [heads,total_q].
      lse=lse.transpose(0,1)
      row=dict(heads=heads,qlens=qlens,zero_query=zero,left=left,packed=packed,rebased=rebased,
       base=base.tolist(),max_abs_ref=float((output.float()-ref).abs().max()),
       mean_abs_ref=float((output.float()-ref).abs().mean()),nonfinite_lse=int((~lse.isfinite()).sum()),
       max_lse_error=float((lse-reflse).abs().max()))
      rows.append(row);print(row,flush=True)
torch.cuda.synchronize()
doc=dict(scope='SM86 BF16 correctness only; full and sliding controls; exact-origin wrapper is a NEW fixture revision, not scored parent parity',
 seed=3751381,device=torch.cuda.get_device_name(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 installed_wrapper_sha256=hashlib.sha256(Path(inspect.getsourcefile(fa)).read_bytes()).hexdigest(),results=rows)
(HERE/'attention_oracle_v2.json').write_text(json.dumps(doc,indent=2))
good=[r for r in rows if r['left']<0 or r['rebased']]
assert all(r['nonfinite_lse']==0 and r['max_abs_ref']<.0003 and r['max_lse_error']<.0005 for r in good)
print('PASS: full attention and exact-origin sliding-window reference; original sliding failure retained.')
