"""Research-only GPU alignment wrapper; arithmetic geometry stays unchanged.
Not enabled in any candidate. B multicast additionally requires WS protocol gates.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _expand_experts(Experts, Padded, Expanded,
                    CAPACITY: tl.constexpr, BLOCK_M: tl.constexpr,
                    CLUSTER: tl.constexpr, TILE: tl.constexpr):
    lane = tl.program_id(0) * TILE + tl.arange(0, TILE)
    active = tl.load(Padded) // BLOCK_M
    # Do not read uninitialized tail expert IDs from the upstream align kernel.
    expert = tl.load(Experts + lane // CLUSTER, lane < active, other=-1)
    tl.store(Expanded + lane, expert, lane < CAPACITY)


def align_for_cluster(local_ids, block_m, num_experts, cluster_size, align_impl):
    if cluster_size not in (2, 4) or type(block_m) is not int or block_m <= 0 or block_m % 4:
        raise ValueError("Unsupported research cluster geometry")
    if local_ids.dtype not in (torch.int32, torch.int64) or not local_ids.is_cuda:
        raise ValueError("CUDA int32/int64 routes required")
    if not local_ids.is_contiguous() or local_ids.ndim != 2:
        raise ValueError("Contiguous [tokens, topk] routes required")
    if type(num_experts) is not int or num_experts <= 0:
        raise ValueError("Positive expert count required")
    quantum = block_m * cluster_size
    sorted_ids, cluster_experts, padded = align_impl(
        local_ids, quantum, num_experts, ignore_invalid_expert=True)
    capacity = cluster_experts.numel() * cluster_size
    expanded = torch.empty((capacity,), device=local_ids.device, dtype=torch.int32)
    if capacity:
        _expand_experts[(triton.cdiv(capacity, 256),)](
            cluster_experts, padded, expanded, capacity, block_m, cluster_size, 256)
    return sorted_ids, expanded, padded
