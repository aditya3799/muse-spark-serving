"""Rank-two W2 feature ownership aligned to the scalar RMS reduction tree."""
import torch
from finalize import local_squares,local_owner_features


def indices(h,device="cpu"):
    # The scored full H4096 CUDA reduction maps eight consecutive values to
    # each thread, and 32 consecutive threads to each of eight warps.
    if h!=4096:
        raise ValueError("Only the compiled 4096-feature reduction layout is qualified")
    band=32*8
    values=torch.arange(h,device=device).reshape(-1,2,band)
    return tuple(values[:,r,:].reshape(-1).contiguous() for r in (0,1))


def bands_from_full(full):
    shape=full.shape[:-1]
    value=full.reshape(*shape,-1,2,256)
    return tuple(value[...,rank,:].reshape(*shape,-1).contiguous() for rank in (0,1))


def restore(parts):
    shape=parts[0].shape[:-1]
    a,b=parts
    return torch.stack([a.reshape(*shape,-1,256),b.reshape(*shape,-1,256)],dim=-2).reshape(*shape,-1)


def candidate(parts,ids,scores,norm_halves,scale):
    # On distinct ranks replace this FP32 addition by the two-rank all_reduce.
    partial=[local_squares(d,ids,w,scale) for d,w in zip(parts,norm_halves)]
    total=partial[0]+partial[1]
    owned=[local_owner_features(d,ids,scores,w,scale,total) for d,w in zip(parts,norm_halves)]
    # Native deployment exchanges [2,M,H/2] via all_to_all_single first.
    # Reinterleave warp bands within each original EP owner's token payload.
    owners=restore(owned)
    return owners[0]+owners[1],owners,total
