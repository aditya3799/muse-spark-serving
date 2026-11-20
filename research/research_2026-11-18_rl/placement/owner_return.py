"""Return migrated experts' raw BF16 rows to their original EP owner before the native finalize."""
import torch
import torch.distributed as dist


def return_rows(down, global_ids, slots, group):
    """All ranks share global IDs and the agreed current placement.
    This eager reference uses dynamically sized peer buffers. A graph-safe fast path
    would also have to bound or pre-stage sizes without reading device counts on the host.
    """
    rank=dist.get_rank(group)
    world=dist.get_world_size(group)
    per_rank=slots.shape[1]
    assert slots.shape[0]==world and down.shape[:-1]==global_ids.shape and down.is_contiguous()
    device=global_ids.device
    slot_tensor=torch.as_tensor(slots,device=device,dtype=torch.long)
    where=torch.empty(slot_tensor.numel(),device=device,dtype=torch.long)
    where[slot_tensor.reshape(-1)]=torch.arange(world,device=device).repeat_interleave(per_rank)
    all_ids=global_ids.reshape(-1).long()
    live=all_ids>=0
    safe_ids=all_ids.clamp_min(0)
    original=safe_ids//per_rank
    current=where[safe_ids]
    flat=down.view(-1,down.shape[-1])
    recv=[]
    sends=[]
    ops=[]
    for peer in range(world):
        if peer==rank:continue
        global_peer=dist.get_global_rank(group,peer) if group is not None else peer
        to_peer=(live&(current==rank)&(original==peer)).nonzero().flatten()
        from_peer=(live&(current==peer)&(original==rank)).nonzero().flatten()
        if to_peer.numel():
            buf=flat.index_select(0,to_peer)
            sends.append(buf)
            ops.append(dist.P2POp(dist.isend,buf,global_peer,group))
        if from_peer.numel():
            buf=flat.new_empty((from_peer.numel(),flat.shape[1]))
            recv.append((from_peer,buf))
            ops.append(dist.P2POp(dist.irecv,buf,global_peer,group))
    if ops:
        for req in dist.batch_isend_irecv(ops):
            req.wait()
    for indices,buffer in recv:
        flat.index_copy_(0,indices,buffer)
    local=torch.where(live&(original==rank),safe_ids%per_rank,-1).to(torch.int32).view_as(global_ids)
    return down,local,sum(t.numel()*t.element_size() for t in sends)
