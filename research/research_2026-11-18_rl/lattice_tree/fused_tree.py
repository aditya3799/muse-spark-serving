"""Build a DFlash2 verification tree in a single grid, one program per request."""
import torch
import triton
import triton.language as tl

@triton.jit
def _grow(ids,scores,log_mass,anchor,out_ids,out_parents,out_pos,out_mask,inv_temp,log_fork,
          DEPTH:tl.constexpr,K:tl.constexpr,N:tl.constexpr,ROWS:tl.constexpr,MASS:tl.constexpr):
    b=tl.program_id(0)
    row=tl.arange(0,ROWS)
    col=tl.arange(0,K)
    frontier=tl.full((ROWS,K),-float("inf"),tl.float32)
    depth_by_node=tl.full((ROWS,),0,tl.int32)
    fork_by_node=tl.full((ROWS,),0,tl.int32)
    greedy_by_node=tl.full((ROWS,),0,tl.int32)
    mask_by_node=tl.full((ROWS,),0,tl.uint32)
    value=tl.full((),0,tl.float32)
    depth=tl.full((),0,tl.int32)
    choice=tl.full((),0,tl.int32)
    predecessor=tl.full((),0,tl.int32)
    fork=tl.full((),0,tl.int32)
    ancestors=tl.full((),1,tl.uint32)
    token=tl.load(anchor+b)
    for node in range(N):
        offset=b*N+node
        tl.store(out_ids+offset,token)
        tl.store(out_parents+offset,predecessor)
        tl.store(out_pos+offset,depth)
        tl.store(out_mask+offset,ancestors)
        if node+1<N:
            legal=depth<DEPTH
            logits=tl.load(scores+((b*DEPTH+depth)*K+choice)*K+col,mask=legal,other=0.).to(tl.float32)*inv_temp
            logits=tl.where(logits==logits,logits,-float("inf"))
            mx,best=tl.max(logits,0,return_indices=True,return_indices_tie_break_left=True)
            # Equal positive infinities split their conditional mass; a fully invalid row is uniform.
            safe=tl.where(mx==float("inf"),tl.where(logits==mx,0.,-float("inf")),logits)
            safe=tl.where(mx==-float("inf"),0.,safe)
            shifted=safe-tl.max(safe,0)
            edges=shifted-tl.log(tl.sum(tl.exp(shifted),0))
            if MASS:
                mass=tl.load(log_mass+b*DEPTH+depth,mask=legal,other=0.)
                edges+=tl.minimum(0.,tl.where(mass==mass,mass,0.))
            child=tl.where(legal,tl.maximum(-1.e30,value+edges+fork.to(tl.float32)*log_fork),-float("inf"))
            frontier=tl.where(row[:,None]==node,child[None,:],frontier)
            depth_by_node=tl.where(row==node,depth,depth_by_node)
            fork_by_node=tl.where(row==node,fork,fork_by_node)
            greedy_by_node=tl.where(row==node,best,greedy_by_node)
            mask_by_node=tl.where(row==node,ancestors,mask_by_node)
            value,index=tl.max(tl.reshape(frontier,(ROWS*K,)),0,return_indices=True,return_indices_tie_break_left=True)
            predecessor=index//K
            choice=index%K
            frontier=tl.where((row[:,None]==predecessor)&(col[None,:]==choice),-float("inf"),frontier)
            depth=tl.sum(tl.where(row==predecessor,depth_by_node,0),0)+1
            prior_choice=tl.sum(tl.where(row==predecessor,greedy_by_node,0),0)
            fork=tl.sum(tl.where(row==predecessor,fork_by_node,0),0)|(prior_choice!=choice).to(tl.int32)
            ancestors=tl.sum(tl.where(row==predecessor,mask_by_node,tl.full((),0,tl.uint32)),0).to(tl.uint32)|(tl.full((),1,tl.uint32)<<(node+1))
            token=tl.load(ids+(b*DEPTH+depth-1)*K+choice)

def build(candidates,scores,anchors,n,temperature=1.,after_fork=1.,log_mass=None):
    import math
    b,depth,k=candidates.shape
    if not (2<=n<=32 and k>1 and k==triton.next_power_of_2(k) and depth>0 and n<=1+sum(k**d for d in range(1,depth+1))):
        raise ValueError("unsupported selector geometry")
    if not (temperature>0 and 0<after_fork<=1):
        raise ValueError("invalid proposal temperature or fork factor")
    assert scores.shape==(b,depth,k,k) and anchors.shape==(b,)
    assert candidates.is_contiguous() and scores.is_contiguous() and anchors.is_contiguous()
    if log_mass is not None:assert log_mass.shape==(b,depth) and log_mass.is_contiguous()
    device=candidates.device
    ids=torch.empty((b,n),dtype=candidates.dtype,device=device)
    parent=torch.empty((b,n),dtype=torch.int32,device=device)
    positions=torch.empty_like(parent)
    bits=torch.empty_like(parent,dtype=torch.uint32)
    _grow[(b,)](candidates,scores,log_mass,anchors,ids,parent,positions,bits,1./temperature,math.log(after_fork),
                depth,k,n,triton.next_power_of_2(n),log_mass is not None,num_warps=4)
    return ids,parent,positions,bits