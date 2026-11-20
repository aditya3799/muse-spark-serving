"""Output-feature-sharded routed FFN with one BF16 activation gather and FP32 distributed expert normalization."""
import torch
import torch.distributed as dist


def gather_features(local, group):
    parts=[torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts,local.contiguous(),group=group)
    return torch.cat(parts,dim=-1)


def distributed_experts(x,ids,scores,slot,rank_index,group,native_up,native_down,norm,owner_size=None):
    """All group members have the same IDs, stable route order and inputs.
    Each stage callback computes full K and returns BF16 with the original global scales.
    """
    m,k=ids.shape
    valid=slot[ids.long()+1]
    selected=valid>=0
    route=torch.nonzero(selected.reshape(-1),as_tuple=False).flatten()
    expert=valid.reshape(-1)[route].long()
    local_width=norm.numel()
    if route.numel()==0:
        # Replicated global IDs and a common expert range guarantee agreement inside the group.
        return x.new_zeros((2,m,local_width) if owner_size is not None else (m,local_width))
    rows=x.index_select(0,torch.div(route,k,rounding_mode="floor"))
    hid_shard=native_up(rows,expert,rank_index)
    assert hid_shard.dtype==torch.bfloat16
    hidden=gather_features(hid_shard,group)
    down=native_down(hidden,expert,rank_index)
    assert down.dtype==torch.bfloat16
    scaled=down.float()*norm
    sums=scaled.square().sum(dim=-1,keepdim=True)
    dist.all_reduce(sums,group=group)
    inv=torch.rsqrt(sums/(local_width*dist.get_world_size(group))+1e-8)
    weighted=scaled*inv*scores.reshape(-1)[route,None]
    contributions=torch.zeros((m*k,local_width),device=x.device,dtype=torch.float32)
    contributions.index_copy_(0,route,weighted)
    out=contributions.view(m,k,local_width)
    if owner_size is not None:
        # Preserve both original EP ranks’ BF16 rounding boundaries before all-reduce.
        owner=valid.reshape(m,k).clamp_min(0)//owner_size
        acc=torch.zeros((2,m,local_width),device=x.device,dtype=torch.float32)
        for j in range(k):
            for r in range(2):
                acc[r].add_(torch.where((owner[:,j]==r)[:,None],out[:,j],0.))
        return acc.to(torch.bfloat16)
    acc=torch.zeros_like(out[:,0])
    for j in range(k):
        acc.add_(out[:,j])
    return acc.to(torch.bfloat16)


def restore_original_rank(parts, group):
    """Transpose [destination EP rank, tokens, my features] to one full-width original EP partial."""
    received = torch.empty_like(parts)
    dist.all_to_all_single(received, parts.contiguous(), group=group)
    return torch.cat(received.unbind(0), dim=-1)
