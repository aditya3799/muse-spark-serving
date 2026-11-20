"""Exact post-gather first-rank selection; native softcap/payload/collective unchanged."""
import torch
import triton
import triton.language as tl

@triton.jit
def select_target(PAIRS, OUT, N: tl.constexpr, TP: tl.constexpr, RT: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rank = tl.arange(0, RT)
    valid = (rank[:, None] < TP) & (token[None, :] < N)
    values = tl.load(PAIRS + 2 * (rank[:, None] * N + token[None, :]), valid, other=0)
    nan = (values != values) & valid
    first_nan = tl.min(tl.where(nan, rank[:, None], TP), axis=0)
    maximum = tl.max(tl.where(valid & ~nan, values, float('-inf')), axis=0)
    first_max = tl.min(tl.where(valid & (values == maximum[None, :]), rank[:, None], TP), axis=0)
    best = tl.where(first_nan < TP, first_nan, first_max)
    ids = tl.load(PAIRS + 2 * (best * N + token) + 1, token < N, other=0)
    # Match the original float payload -> long conversion, including its rounding.
    tl.store(OUT + token, ids.to(tl.int64), token < N)

def native_select(pairs):
    best_rank = pairs[..., 0].argmax(dim=0, keepdim=True)
    return pairs[..., 1].gather(0, best_rank).squeeze(0).long()

def fused_select(pairs):
    if (not pairs.is_cuda or pairs.dtype != torch.float32 or pairs.ndim != 3 or
            pairs.shape[2] != 2 or not pairs.is_contiguous() or
            not 1 <= pairs.shape[0] <= 16 or pairs.shape[1] < 1):
        return native_select(pairs)
    tp, n, _ = pairs.shape
    with torch.cuda.device(pairs.device):
        out = torch.empty(n, dtype=torch.int64, device=pairs.device)
        select_target[(triton.cdiv(n, 32),)](pairs, out, n, tp, triton.next_power_of_2(tp), 32, num_warps=4)
    return out
