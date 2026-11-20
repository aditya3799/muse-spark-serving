def candidate_topk(
    hidden: torch.Tensor,
    lm_head: nn.Module,
    k: int,
    *,
    with_partition: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Global top-k candidates from the target head: hidden [N, H] -> ids / vals [N, K].

    Under TP (vocab-sharded lm_head): local top-k per shard, all-gather K logits/ids (not
    the full vocab), then a global top-k -- identical candidates at O(tp*K) instead of
    O(vocab) gather bandwidth. with_partition also returns the full-vocabulary lse [N] fp32.
    """
    # The worker screens the head before capture, but the eager fallbacks attach
    # whatever the target has.
    weight = getattr(lm_head, "weight", None)
    quant_method = getattr(lm_head, "quant_method", None)
    use_quant_head = should_apply_lm_head_quant_method(lm_head, quant_method)
    if not use_quant_head and not is_dense_head_weight(weight):
        raise RuntimeError(
            "DFlash candidate top-k requires a dense FP16/BF16/FP32 target lm_head "
            "or a supported lm_head.quant_method."
        )
    tp = get_parallel().tp_size == 1
    num_org = (
        int(lm_head.org_vocab_size)
        if tp
        else int(lm_head.shard_indices.num_org_elements)
    )
    logits = _project_candidate_logits(
        hidden, lm_head, num_org=num_org, use_quant_head=use_quant_head
    )
    lse = None
    if with_partition:
        vals, ids, lse = lilicorr_topk_lse(logits, k)
    else:
        vals, ids = _radix_topk(logits, k)
    if tp:
        return ids.long(), vals, lse
    global_ids = ids.long() + int(lm_head.shard_indices.org_vocab_start_index)
    if int(lm_head.org_vocab_size) <= (1 << 24):
        # All valid token IDs are exactly representable in FP32. Keep values
        # and IDs in separate leading planes, preserving contiguous top-k rows.
        packed = torch.stack((vals.float(), global_ids.float()), dim=0)
        gathered = tensor_model_parallel_all_gather(packed, dim=-1)
        gathered_vals, gathered_ids = gathered[0], gathered[1]
    else:
        gathered_vals = tensor_model_parallel_all_gather(vals.float(), dim=-1)
        gathered_ids = tensor_model_parallel_all_gather(global_ids, dim=-1)
    top_vals, sel = torch.topk(gathered_vals, k, dim=-1)
    if lse is not None:
        lse = torch.logsumexp(
            tensor_model_parallel_all_gather(lse.unsqueeze(-1), dim=-1), dim=-1
        )
    return torch.gather(gathered_ids, -1, sel).long(), top_vals, lse
