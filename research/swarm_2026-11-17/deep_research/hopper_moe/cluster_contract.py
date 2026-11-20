"""Research-only cluster routing contract. No import-time CUDA work or serving hook."""
from copy import deepcopy


def policy_for_cluster(original, sm_version, cluster_size, num_sms):
    """Fail closed: matching experts is insufficient without inter-CTA credits."""
    if sm_version != 90 or cluster_size not in (2, 4):
        return None, "architecture_or_cluster"
    if not isinstance(num_sms, int) or num_sms <= 0:
        return None, "missing_actual_sm_count"
    if original.get("use_stream_k") is not False or original.get("use_f16_accum") is not False:
        return None, "arithmetic_contract"
    if original.get("use_warp_spec") is not True:
        return None, "non_ws_has_no_remote_stage_release"
    if original.get("use_tma_b") is not True or original.get("use_mbarrier") is not True:
        return None, "missing_tma_barriers"
    if any(original.get(key, False) for key in ("use_tma_a", "use_tma_as", "use_tma_c")):
        return None, "indexed_contract"
    if original.get("multi_cast_size_a", 1) != 1:
        return None, "mixed_cluster_axes"
    block, warp = original.get("block_shape", ()), original.get("warp_shape", ())
    if len(block) != 3 or len(warp) != 3 or any(type(x) is not int or x <= 0 for x in (*block, *warp)):
        return None, "invalid_geometry"
    if any(b % w for b, w in zip(block, warp)) or block[0] % 4:
        return None, "invalid_geometry"
    if original.get("num_stages", 0) < 3:
        return None, "unqualified_pipeline_depth"
    ctas_per_sm = original.get("num_ctas_per_sm", 1)
    if type(ctas_per_sm) is not int or ctas_per_sm <= 0 or ctas_per_sm * num_sms % cluster_size:
        return None, "grid_not_cluster_multiple"
    result = deepcopy(original)
    result["multi_cast_size_b"] = cluster_size
    for key in ("block_shape", "warp_shape", "num_stages", "use_f16_accum", "use_stream_k"):
        assert result[key] == original[key]
    return result, "research_only_runtime_unqualified"


def align_reference(routes, num_experts, block_m, cluster_size):
    """Stable CPU oracle. Invalid negative routes are omitted; duplicates kept."""
    assert type(num_experts) is int and num_experts > 0
    assert type(block_m) is int and block_m > 0
    assert cluster_size in (1, 2, 4)
    assert all(type(e) is int and -1 <= e < num_experts for e in routes)
    quantum = block_m * cluster_size
    ids, blocks = [], []
    sentinel = len(routes)
    for expert in range(num_experts):
        owned = [i for i, e in enumerate(routes) if e == expert]
        if not owned:
            continue
        padded = (len(owned) + quantum - 1) // quantum * quantum
        ids.extend(owned + [sentinel] * (padded - len(owned)))
        blocks.extend([expert] * (padded // block_m))
    return ids, blocks, len(ids)


def cost_reference(counts, block_m, cluster_size):
    """Counts issued B tiles and math CTAs, not time or measured HBM bytes."""
    base = sum((n + block_m - 1) // block_m for n in counts)
    clusters = sum((n + block_m * cluster_size - 1) // (block_m * cluster_size) for n in counts)
    return {
        "baseline_row_tiles": base,
        "candidate_row_tiles": clusters * cluster_size,
        "baseline_B_loads_per_NK_tile": base,
        "candidate_B_loads_per_NK_tile": clusters,
        "extra_padded_rows": (clusters * cluster_size - base) * block_m,
        "B_issue_reduction_ratio": base / clusters if clusters else 1.0,
        "math_tile_inflation": clusters * cluster_size / base if base else 1.0,
    }
