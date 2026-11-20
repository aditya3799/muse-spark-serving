"""Research-only owner for captured prefill auxiliary output.

The owner lives outside graph pools. Host leases prevent overlapping producer
calls; CUDA events order a different producer/consumer stream without a host
synchronize. Event operations are outside captured transformer bodies.
"""
from contextlib import contextmanager

import torch


class PrefillAuxOutputOwner:
    def __init__(self, max_tokens, hidden_size, capture_ids, device):
        self.capture_ids = tuple(capture_ids)
        self.hidden_size = hidden_size
        self.buffer = torch.empty(
            (max_tokens, hidden_size * len(self.capture_ids)),
            dtype=torch.bfloat16,
            device=device,
        )
        self._state = "idle"
        self._generation = 0
        self._ready = torch.cuda.Event()
        self._consumed = torch.cuda.Event()
        self._have_consumed = False

    def view(self, rows):
        if not 0 < rows <= self.buffer.shape[0]:
            raise ValueError("aux output exceeds owner capacity")
        return self.buffer[:rows]

    def owns(self, output):
        return (
            torch.is_tensor(output)
            and output.ndim == 2
            and 0 < output.shape[0] <= self.buffer.shape[0]
            and output.shape[1] == self.buffer.shape[1]
            and output.dtype == self.buffer.dtype
            and output.device == self.buffer.device
            and output.stride() == self.buffer.stride()
            and output.data_ptr() == self.buffer.data_ptr()
        )

    def adopt(self, output, rows):
        if self.owns(output) and output.shape[0] <= rows <= self.buffer.shape[0]:
            return self.view(rows)
        return None

    def begin_produce(self):
        if self._state != "idle":
            raise RuntimeError("prior auxiliary consumer has not released its lease")
        stream = torch.cuda.current_stream(self.buffer.device)
        if self._have_consumed:
            stream.wait_event(self._consumed)
        self._state = "producing"
        self._generation += 1
        return self._generation

    def end_produce(self, generation, output):
        if self._state != "producing" or generation != self._generation:
            raise RuntimeError("stale auxiliary producer lease")
        if not self.owns(output):
            raise RuntimeError("captured prefill did not return its owned auxiliary output")
        self._ready.record(torch.cuda.current_stream(self.buffer.device))
        self._state = "ready"

    def cancel_produce(self, generation):
        # Preserve ordering of any already-enqueued writes after an exception.
        if self._state == "producing" and generation == self._generation:
            self._consumed.record(torch.cuda.current_stream(self.buffer.device))
            self._have_consumed = True
            self._state = "idle"

    @contextmanager
    def consume(self, output):
        if not self.owns(output):
            yield
            return
        if self._state != "ready":
            raise RuntimeError("auxiliary output is not ready for its consumer")
        stream = torch.cuda.current_stream(self.buffer.device)
        stream.wait_event(self._ready)
        self._state = "consuming"
        try:
            yield
        finally:
            self._consumed.record(stream)
            self._have_consumed = True
            self._state = "idle"


class DirectAuxCapture:
    """Same FP32-to-BF16 conversion, written into the final feature slices."""

    def __init__(self, buffer, hidden_size, capture_ids):
        self.buffer = buffer
        self.hidden_size = hidden_size
        self.capture_ids = tuple(capture_ids)
        self.index = 0

    def capture(self, layer_idx, residual):
        if self.index >= len(self.capture_ids) or layer_idx != self.capture_ids[self.index]:
            raise RuntimeError("auxiliary capture order changed")
        if (
            residual.shape != (self.buffer.shape[0], self.hidden_size)
            or residual.dtype != torch.float32
            or residual.device != self.buffer.device
        ):
            raise RuntimeError("auxiliary residual metadata changed")
        start = self.index * self.hidden_size
        self.buffer[:, start : start + self.hidden_size].copy_(residual)
        self.index += 1

    def finalize(self):
        if self.index != len(self.capture_ids):
            raise RuntimeError("incomplete auxiliary capture")
        return self.buffer


def direct_aux_capture(buffer, residual, capture_ids, operator_dtype, sp_active):
    ids = tuple(capture_ids)
    if (
        buffer is None
        or sp_active
        or operator_dtype != torch.bfloat16
        or residual.dtype != torch.float32
        or residual.ndim != 2
        or not ids
        or any(type(i) is not int or i < 0 for i in ids)
        or ids != tuple(sorted(set(ids)))
        or buffer.shape != (residual.shape[0], residual.shape[1] * len(ids))
        or buffer.dtype != torch.bfloat16
        or buffer.device != residual.device
        or not buffer.is_contiguous()
    ):
        return None
    return DirectAuxCapture(buffer, residual.shape[1], ids)
