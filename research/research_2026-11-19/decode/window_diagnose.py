"""Diagnose native SWA rebasing against explicit FP32 attention, outside timed promotion."""
import hashlib
import json
import argparse
from pathlib import Path
import torch
import sgl_kernel.flash_attn as fa

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--splits',type=int,default=0);a=p.parse_args()
torch.manual_seed(842671)
dev="cuda";b=2;n=8;heads=16;kvheads=2;d=64;cap=8192;left=2047
k=torch.randn((16000,1,kvheads,d),device=dev,dtype=torch.bfloat16)*.35
v=torch.randn_like(k)*.35
q=torch.randn((b*n,heads,d),device=dev,dtype=torch.bfloat16)*.31
pt=torch.stack([torch.randperm(7900,device=dev)+100,torch.randperm(7900,device=dev)+8000]).int()
pt=torch.nn.functional.pad(pt,(0,cap-7900))
l=torch.tensor([2853,7171],device=dev,dtype=torch.int32)
cu=torch.arange(b+1,device=dev,dtype=torch.int32)*n
ref=[]
for row in range(b):
    length=int(l[row]);query=q[row*n:(row+1)*n].float().transpose(0,1)
    keys=k[pt[row,:length].long(),0].float().repeat_interleave(heads//kvheads,dim=1).permute(1,2,0)
    values=v[pt[row,:length].long(),0].float().repeat_interleave(heads//kvheads,dim=1).transpose(0,1)
    score=query@keys/(d**.5)
    pos=torch.arange(length,device=dev)[None,:]
    qpos=torch.arange(n,device=dev)[:,None]+length-n
    mask=(pos<=qpos)&(pos>=qpos-left)
    score.masked_fill_(~mask[None,:,:],-float("inf"))
    ref.append((score.softmax(-1)@values).transpose(0,1))
reference=torch.cat(ref)
out={"scope":"One native SM86 BF16 SWA call; diagnostic, not performance evidence", "arms":{}}
baseline=None
for align in [0,64,128,1920]:
    if align:
        base=((l-left-n).clamp_min(0)//align)*align
        lens=l-base
        width=left+n+align-1
        cols=base[:,None]+torch.arange(width,device=dev)[None,:]
        table=pt.gather(1,cols.long().clamp_max(cap-1)).contiguous()
    else:
        base=torch.zeros_like(l);lens=l;table=pt;width=cap
    for sched in [False,True]:
        plan=fa.get_scheduler_metadata(batch_size=b,max_seqlen_q=n,max_seqlen_k=width,
            num_heads=heads,num_heads_k=kvheads,headdim=d,headdim_v=d,cache_seqlens=lens,
            qkv_dtype=torch.bfloat16,cu_seqlens_q=cu,page_size=1,causal=True,
            window_size=(left,0),num_splits=a.splits) if sched else None
        result,lse,*_=fa.flash_attn_with_kvcache(q,k,v,cache_seqlens=lens,page_table=table,
            cu_seqlens_q=cu,max_seqlen_q=n,causal=True,window_size=(left,0),num_splits=a.splits,
            scheduler_metadata=plan,return_softmax_lse=True,ver=3)
        torch.cuda.synchronize()
        if baseline is None:baseline=result.clone()
        out["arms"][f"align{align}_schedule{int(sched)}"]=dict(
            base=base.tolist(),lengths=lens.tolist(),width=width,
            max_abs_ref=float((result.float()-reference).abs().max()),
            mean_abs_ref=float((result.float()-reference).abs().mean()),
            max_abs_parent=float((result.float()-baseline.float()).abs().max()),
            wrong=int((result.view(torch.int16)!=baseline.view(torch.int16)).sum()),
            plan=None if plan is None else plan.tolist(),
            first=result.flatten()[:8].tolist(),lse_min=float(lse.min()),lse_max=float(lse.max()))
        print(f"align{align} schedule{int(sched)}",out["arms"][f"align{align}_schedule{int(sched)}"],flush=True)
(HERE/f"window_diagnostic_splits{a.splits}.json").write_text(json.dumps(out,indent=2))
