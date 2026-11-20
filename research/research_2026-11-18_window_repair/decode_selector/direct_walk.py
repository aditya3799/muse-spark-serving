import torch
import triton
import triton.language as tl


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    uniforms_ptr,
    temperatures_ptr,
    greedy_ptr,
    tokens_ptr,
    q_ptr,
    slots: tl.constexpr,
    top_k: tl.constexpr,
    candidate_out_ptr=None,
):
    """One program per request: a slot's K scores stay in registers and the walk is a
    loop, so the slot-to-slot dependency costs nothing instead of one kernel each."""
    row = tl.program_id(0)
    offsets = tl.arange(0, top_k)
    temperature = tl.load(temperatures_ptr + row)
    greedy = tl.load(greedy_ptr + row) != 0
    previous = 0
    for slot in range(slots):
        base = (row * slots + slot) * top_k
        scores = tl.load(scores_ptr + (base + previous) * top_k + offsets).to(
            tl.float32
        )
        if greedy:
            # Same pick as torch.argmax: the first NaN if any, else the first max.
            is_nan = scores != scores
            best = tl.max(tl.where(is_nan, -float("inf"), scores), axis=0)
            has_nan = tl.max(is_nan.to(tl.int32), axis=0) > 0
            hit = tl.where(has_nan, is_nan, scores == best)
            index = tl.min(tl.where(hit, offsets, top_k), axis=0)
            probabilities = tl.where(offsets == index, 1.0, 0.0)
        else:
            scaled = scores / temperature
            exponentials = tl.exp(scaled - tl.max(scaled, axis=0))
            probabilities = exponentials / tl.sum(exponentials, axis=0)
            uniform = tl.load(uniforms_ptr + row * slots + slot)
            index = tl.sum(
                tl.where(uniform >= tl.cumsum(probabilities, axis=0), 1, 0), axis=0
            )
            index = tl.minimum(index, top_k - 1)
        if candidate_out_ptr is not None:
            tl.store(candidate_out_ptr + base + offsets, tl.load(candidate_ptr + base + offsets))
        tl.store(q_ptr + base + offsets, probabilities)
        tl.store(tokens_ptr + row * slots + slot, tl.load(candidate_ptr + base + index))
        previous = index

def selector_walk_triton(
    *,
    candidate_ids,
    scores,
    uniforms,
    temperatures,
    greedy_mask,
    tokens=None,
    q_rows=None,
    candidate_out=None,
):
    batch, slots, top_k = candidate_ids.shape
    if tokens is None:
        tokens = torch.empty((batch, slots), dtype=torch.int64, device=scores.device)
    if q_rows is None:
        q_rows = torch.empty(
            (batch, slots, top_k), dtype=torch.float32, device=scores.device
        )
    _selector_walk_kernel[(batch,)](
        scores.contiguous(),
        candidate_ids.contiguous(),
        uniforms.contiguous(),
        temperatures.contiguous(),
        greedy_mask.contiguous(),
        tokens,
        q_rows,
        slots=slots,
        top_k=top_k,
        candidate_out_ptr=candidate_out,
        num_warps=1,
    )
    return tokens, q_rows
