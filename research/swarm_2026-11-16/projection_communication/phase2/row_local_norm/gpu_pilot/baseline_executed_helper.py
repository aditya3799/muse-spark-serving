def prefill_row_projection(group, x, weight):
    """Compute full output rows once, then gather in original token order.

    Input and weights must be replicated on the TP group. The complete K
    reduction and full output dimension remain local, with FP32 output before
    the single BF16 rounding. No persistent repacked weight or padding exists.
    """
    m_local = x.shape[0] // group.world_size
    start = group.rank_in_group * m_local
    local_fp32 = torch.mm(x[start:start + m_local], weight.t(), out_dtype=torch.float32)
    local = local_fp32.to(x.dtype)
    del local_fp32
    return group.all_gather(local, dim=0)