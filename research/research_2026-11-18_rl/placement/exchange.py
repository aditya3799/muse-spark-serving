"""Apply pre-agreed whole-expert exchanges while workers are idle and update the existing lookup in place."""
from hashlib import sha256
import numpy as np
import torch
import torch.distributed as dist
from placement import NATIVE, local_map

TENSORS=tuple(f"{stage}_{component}" for stage in ("w13","w2") for component in ("weight","weight_scale","post_scale"))


def swap_slots(module, group, actions, before=NATIVE):
    """All ranks enter after draining readers. On communication failure the caller must stop the worker."""
    rank=dist.get_rank(group)
    size=dist.get_world_size(group)
    slots=np.array(before,dtype=np.int32,copy=True)
    actions=tuple(tuple(map(int,a)) for a in actions)
    bank=module._humming_layer
    parts=[getattr(bank,name) for name in TENSORS]
    if size!=len(slots) or any(t.shape[0]!=slots.shape[1] or not t.is_contiguous() for t in parts):
        raise ValueError("Unexpected Humming topology or expert tensor layout")
    for src,i,dst,j in actions:
        if src==dst or not(0<=src<size and 0<=dst<size and 0<=i<slots.shape[1] and 0<=j<slots.shape[1]):
            raise ValueError("Invalid exchange endpoints")
    dev=parts[0].device
    # All peers verify the same plan and starting ownership *before* touching any weight.
    digest=sha256(slots.tobytes()+np.asarray(actions,dtype=np.int64).tobytes()).digest()
    tag=torch.tensor(list(digest),dtype=torch.uint8,device=dev)
    copies=[torch.empty_like(tag) for _ in range(size)]
    dist.all_gather(copies,tag,group=group)
    if any(not torch.equal(t,tag) for t in copies):
        raise RuntimeError("Expert plans disagree across ranks")
    widths=[t[0].numel()*t.element_size() for t in parts]
    outgoing=torch.empty(sum(widths),dtype=torch.uint8,device=dev)
    incoming=torch.empty_like(outgoing)
    with torch.no_grad():
        for src,i,dst,j in actions:
            slots[src,i],slots[dst,j]=slots[dst,j],slots[src,i]
            if rank not in (src,dst):
                continue
            peer,slot=(dst,i) if rank==src else (src,j)
            global_peer=dist.get_global_rank(group,peer) if group is not None else peer
            views=[t[slot:slot+1].reshape(-1).view(torch.uint8) for t in parts]
            for dest,source in zip(outgoing.split(widths),views):
                dest.copy_(source)
            ops=[dist.P2POp(dist.isend,outgoing,global_peer,group),dist.P2POp(dist.irecv,incoming,global_peer,group)]
            for req in dist.batch_isend_irecv(ops):
                req.wait()
            for dest,source in zip(views,incoming.split(widths)):
                dest.copy_(source)
        # A graph observes the same storage addresses with the new complete map.
        dist.barrier(group=group)
        mapping=np.full(slots.size,-1,dtype=np.int32)
        mapping[slots[rank]]=np.arange(slots.shape[1],dtype=np.int32)
        module.update_expert_map_(torch.as_tensor(mapping,device=dev))
        dist.barrier(group=group)
    return slots
