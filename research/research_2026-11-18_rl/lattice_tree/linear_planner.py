"""Greedy-only linear DFlash planner with a discounted expected-commit objective."""
import triton
import triton.language as tl


@triton.jit
def _linear(scores,ids,out,inverse_temperature,gamma,SLOTS:tl.constexpr,K:tl.constexpr):
    b=tl.program_id(0)
    col=tl.arange(0,K)
    row=tl.arange(0,K)
    slot_ids=tl.arange(0,triton.next_power_of_2(SLOTS))
    actions=tl.full((triton.next_power_of_2(SLOTS),K),0,tl.int32)
    value=tl.full((K,),0.,tl.float32)
    for i in range(SLOTS):
        j=SLOTS-1-i
        block=(b*SLOTS+j)*K*K
        x=tl.load(scores+block+row[:,None]*K+col[None,:]).to(tl.float32)
        missing=x!=x
        finite=tl.where(missing,-float("inf"),x)
        peak=tl.max(finite,1)
        logits=tl.where(peak[:,None]==float("inf"),tl.where(finite==float("inf"),0.,-float("inf")),finite)
        logits=tl.where(peak[:,None]==-float("inf"),0.,logits)
        p=tl.exp((logits-tl.max(logits,1)[:,None])*inverse_temperature)
        p=p/tl.sum(p,1)[:,None]
        utilities=p*(1.+gamma*value[None,:])
        value,choice=tl.max(utilities,1,return_indices=True,return_indices_tie_break_left=True)
        # Retain the scored parent's first-NaN policy at an invalid row.
        first_nan=tl.min(tl.where(missing,col[None,:],K),1)
        choice=tl.where(first_nan<K,first_nan,choice)
        value=tl.where(first_nan<K,0.,value)
        actions=tl.where(slot_ids[:,None]==j,choice[None,:],actions)
    predecessor=tl.full((),0,tl.int32)
    for j in range(SLOTS):
        choices=tl.sum(tl.where(slot_ids[:,None]==j,actions,0),0)
        predecessor=tl.sum(tl.where(col==predecessor,choices,0),0)
        token=tl.load(ids+(b*SLOTS+j)*K+predecessor)
        tl.store(out+b*SLOTS+j,token)
