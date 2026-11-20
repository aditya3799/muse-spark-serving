"""CPU prototype: select a configuration by expected EP-local route count.

This selects tuning only. Kernel valid_shape_m must remain the actual route
array length and alignment must use the selected stage block size.
"""
from copy import deepcopy

def select_ep_config(intervals, actual_routes, local_experts, global_experts):
    if not 0 < local_experts <= global_experts or global_experts % local_experts:
        raise ValueError("Expected a positive equal-size expert partition")
    if actual_routes < 0:
        raise ValueError("Negative route count")
    if actual_routes == 0:
        return None
    tune_m = (actual_routes * local_experts + global_experts - 1) // global_experts
    config = next(deepcopy(c) for lo, hi, c in intervals if lo < tune_m <= hi)
    if config.get('use_stream_k') is not False or config.get('use_f16_accum') is not False:
        raise ValueError("Expected owned FP32, non-Stream-K tuning intervals")
    return dict(tune_m=tune_m, valid_shape_m=actual_routes, config=config)
