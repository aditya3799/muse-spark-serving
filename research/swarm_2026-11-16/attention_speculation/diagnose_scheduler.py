import torch
from sgl_kernel.flash_attn import get_scheduler_metadata, flash_attn_with_kvcache
for page in (1,256):
    b,n,hq,hk,d,qlen=1,8192,16,2,64,8
    q=torch.randn(b*qlen,hq,d,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(b*n//page,page,hk,d,device='cuda',dtype=q.dtype)
    v=torch.randn_like(k)
    lens=torch.full((b,),25,device='cuda',dtype=torch.int32)
    cq=torch.tensor([0,qlen],device='cuda',dtype=torch.int32)
    ck=torch.tensor([0,25],device='cuda',dtype=torch.int32)
    table=torch.arange(b*n//page,device='cuda',dtype=torch.int32).view(b,-1)
    for maxk in (25,8192):
        for use_new in (False,True):
            skw=dict(batch_size=b,max_seqlen_q=qlen,max_seqlen_k=maxk,num_heads=hq,num_heads_k=hk,headdim=d,cache_seqlens=lens,qkv_dtype=q.dtype,cu_seqlens_q=cq,page_size=page,causal=True)
            if use_new: skw['cu_seqlens_k_new']=ck
            sched=get_scheduler_metadata(**skw)
            try:
                flash_attn_with_kvcache(q=q,k_cache=k,v_cache=v,page_table=table,cache_seqlens=lens,cu_seqlens_q=cq,cu_seqlens_k_new=ck,max_seqlen_q=qlen,causal=True,scheduler_metadata=sched)
                status='ok'
            except Exception as exc:status=str(exc)
            print(dict(page=page,maxk=maxk,cu_new=use_new,shape=list(sched.shape),status=status),flush=True)
