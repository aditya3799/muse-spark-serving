"""Greedy packed-chain verification sidecar for static-capacity CUDA graph replay.

All inputs are contiguous device tensors. Starts is the exclusive prefix sum of
scheduled widths (1..stride); prefix is the previous committed sequence length.
Cache slots are allocated by the caller and are indexed in target input order.
Returned source/slot pairs cover exactly the accepted target input prefix.
"""
from dataclasses import dataclass
import torch
import triton
import triton.language as tl


@triton.jit
def _accept(ids, predicted, starts, prefix, slots, lengths, bonus, new_prefix,
            emitted, sources, retained, STRIDE: tl.constexpr):
    request = tl.program_id(0)
    col = tl.arange(0, STRIDE)
    start = tl.load(starts + request)
    width = tl.load(starts + request + 1) - start
    pred = tl.load(predicted + start + col, mask=col < width, other=0)
    draft = tl.load(ids + start + col + 1, mask=col + 1 < width, other=0)
    stop = tl.min(tl.where((col + 1 < width) & (draft != pred), col, width - 1), 0)
    live = col <= stop
    slot = tl.load(slots + start + col, mask=live, other=-1)
    tl.store(lengths + request, stop + 1)
    tl.store(new_prefix + request, tl.load(prefix + request) + stop + 1)
    tl.store(bonus + request, tl.load(predicted + start + stop))
    out = request * STRIDE + col
    tl.store(emitted + out, tl.where(live, pred, 0))
    tl.store(sources + out, tl.where(live, start + col, -1))
    tl.store(retained + out, slot)


@dataclass
class GreedyCommit:
    lengths: torch.Tensor
    bonus: torch.Tensor
    new_prefix: torch.Tensor
    emitted: torch.Tensor
    sources: torch.Tensor
    retained_slots: torch.Tensor

    @classmethod
    def allocate(cls, requests, stride, device):
        if stride < 1 or stride & (stride - 1):
            raise ValueError("output stride must be a positive power of two")
        def alloc(shape, dtype):
            return torch.empty(shape, device=device, dtype=dtype)
        return cls(alloc(requests, torch.int32), alloc(requests, torch.int64),
                   alloc(requests, torch.int64), alloc((requests, stride), torch.int64),
                   alloc((requests, stride), torch.int32), alloc((requests, stride), torch.int64))

    def verify(self, ids, predicted, starts, prefix, slots):
        requests, stride = self.emitted.shape
        _accept[(requests,)](ids, predicted, starts, prefix, slots, self.lengths, self.bonus,
                            self.new_prefix, self.emitted, self.sources, self.retained_slots,
                            STRIDE=stride, num_warps=1)

