"""Choose an FFN data layout from its alignment tiles and scratch requirement."""


def dense_route_tiles(route_slots, global_experts, stages):
    if type(global_experts) is not int or global_experts <= 0:
        return False
    heavy_rows = max(stages[stage]["block_shape"][0] for stage in ("w13", "w2"))
    return route_slots > global_experts * heavy_rows


def route_regime(route_slots, local_experts, global_experts, plan):
    if plan is None or route_slots <= 0 or local_experts <= 0 or global_experts < local_experts:
        return "indexed"
    # Once even the mean global-expert load fills a heavy MMA tile, keep the
    # existing indexed layout and fuse the scale and gate work in place.
    if dense_route_tiles(route_slots, global_experts, plan["heavy"]):
        return "indexed_large"
    # Worst-case extra arena for experts with <= cutoff routes. The packed
    # candidate uses the identical three BF16 widths per route as the parent.
    light_arena = min(route_slots, local_experts * plan["cutoff"])
    if 2 * light_arena <= route_slots:
        return "grouped"
    return "indexed"
