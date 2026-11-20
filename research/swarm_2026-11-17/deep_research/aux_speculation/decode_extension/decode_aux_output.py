"""Lease an existing decode runner's canonical auxiliary output buffer."""
import torch
from sglang.srt.layers.prefill_aux_output import PrefillAuxOutputOwner


class DecodeAuxOutputLease(PrefillAuxOutputOwner):
    def __init__(self, buffer, hidden_size, capture_ids):
        ids = tuple(capture_ids)
        if (not torch.is_tensor(buffer) or buffer.ndim != 2
            or buffer.shape[0] <= 0 or hidden_size <= 0
            or not ids or any(type(i) is not int or i < 0 for i in ids)
            or ids != tuple(sorted(set(ids)))
            or buffer.shape[1] != hidden_size * len(ids)
            or buffer.dtype != torch.bfloat16 or not buffer.is_contiguous()):
            raise ValueError("invalid decode auxiliary output buffer")
        # The runner/global input-buffer pool remains the allocation owner.
        # This object retains and leases that exact storage; it allocates none.
        self.buffer = buffer
        self.hidden_size = hidden_size
        self.capture_ids = ids
        self._state = "idle"
        self._generation = 0
        self._producer_stream = None
        self._consumer_stream = torch.cuda.current_stream(buffer.device)
        self._recorded_streams = {self._consumer_stream}

    def validate_binding(self, buffer, spec):
        if (spec != (self.hidden_size, self.capture_ids)
            or not self.owns(buffer) or buffer.shape != self.buffer.shape):
            raise RuntimeError("decode auxiliary binding changed; rebuild captured runner")
