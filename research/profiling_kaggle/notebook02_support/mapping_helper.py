def _map_topk_ids_to_local_experts(
    topk_ids: torch.Tensor, local_expert_mapping: torch.Tensor
) -> torch.Tensor:
    """Map global expert ids while preserving the ``-1`` drop sentinel."""
    return torch.where(
        topk_ids < 0,
        -1,
        local_expert_mapping[topk_ids.clamp_min(0)],
    )
