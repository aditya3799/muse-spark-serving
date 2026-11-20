"""Isolated, opt-in DFlash phase recorder; never synchronizes CUDA.

Install only after warm-up in a research process. Values measure elapsed time on
one stream, including idle gaps. They are not GPU-active time or cross-rank time.
No tensor contents, prompts, tokens, or request identifiers are recorded.
"""
import contextlib
import functools
import time
from collections import deque

import torch


class PhaseRecorder:
    def __init__(self, *, capacity=16, max_phases=20, max_requests=64,
                 every=32, max_records=256, device="cuda"):
        if min(capacity, max_phases, max_requests, every, max_records) < 1:
            raise ValueError("all recorder limits must be positive")
        self.every = every
        self.max_records = max_records
        self.max_requests = max_requests
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.max_phases = max_phases
        self.slots = []
        for _ in range(capacity):
            self.slots.append({
                "events": [torch.cuda.Event(enable_timing=True)
                           for _ in range(2 * max_phases + 2)],
                "commits": torch.empty(max_requests, dtype=torch.int32,
                                       device="cpu", pin_memory=True),
            })
        self.free = deque(range(capacity))
        self.pending = deque()
        self.records = []
        self.active = None
        self.calls = 0
        self.dropped = 0
        self.phase_overflow = 0
        self.capture_aborts = 0
        self._undo = []

    def begin(self, metadata):
        self.calls += 1
        if self.active is not None:
            raise RuntimeError("nested top-level steps are unsupported")
        if (self.calls - 1) % self.every:
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        self.poll()
        if len(self.records) + len(self.pending) >= self.max_records:
            return False
        if not self.free:
            self.dropped += 1
            return False
        index = self.free.popleft()
        slot = self.slots[index]
        slot.update(index=index, metadata=dict(metadata), phases=[],
                    cursor=1, commit_count=0, status="ok",
                    host_start_ns=time.perf_counter_ns())
        slot["events"][0].record()
        self.active = slot
        return True

    @contextlib.contextmanager
    def phase(self, name):
        slot = self.active
        if slot is None:
            yield
            return
        if torch.cuda.is_current_stream_capturing():
            slot["status"] = "nested_capture_invalidates_timing"
            yield
            return
        if len(slot["phases"]) >= self.max_phases:
            self.phase_overflow += 1
            yield
            return
        start, end = slot["cursor"], slot["cursor"] + 1
        slot["cursor"] += 2
        phase = {"name": name, "start": start, "end": end,
                 "host_start_ns": time.perf_counter_ns()}
        slot["phases"].append(phase)
        slot["events"][start].record()
        try:
            yield
        except BaseException:
            slot["status"] = "phase_error"
            raise
        finally:
            if torch.cuda.is_current_stream_capturing():
                phase["capture_invalid"] = True
                slot["status"] = "nested_capture_invalidates_timing"
            else:
                slot["events"][end].record()
            phase["host_end_ns"] = time.perf_counter_ns()

    def end(self, commits=None, status="ok"):
        slot = self.active
        if slot is None:
            return
        if torch.cuda.is_current_stream_capturing():
            self.capture_aborts += 1
            self.free.append(slot["index"])
            self.active = None
            return
        try:
            if commits is not None:
                if commits.dtype != torch.int32 or commits.device != self.device:
                    slot["metadata"]["commit_histogram_omitted"] = "dtype_or_device"
                elif commits.numel() > self.max_requests or commits.ndim != 1:
                    slot["metadata"]["commit_histogram_omitted"] = "shape_limit"
                else:
                    count = commits.numel()
                    slot["commits"][:count].copy_(commits, non_blocking=True)
                    slot["commit_count"] = count
            slot["status"] = status if status != "ok" else slot["status"]
            slot["host_end_ns"] = time.perf_counter_ns()
            slot["done"] = slot["cursor"]
            slot["events"][slot["done"]].record()
            self.pending.append(slot["index"])
        finally:
            self.active = None

    def poll(self):
        if torch.cuda.is_current_stream_capturing():
            return len(self.pending)
        # Scan all completed slots: no assumptions about stream completion order.
        for _ in range(len(self.pending)):
            index = self.pending.popleft()
            slot = self.slots[index]
            done = slot["events"][slot["done"]]
            if not done.query():
                self.pending.append(index)
                continue
            origin = slot["events"][0]
            phases = []
            for p in slot["phases"]:
                if p.get("capture_invalid"):
                    continue
                phases.append({
                    "name": p["name"],
                    "stream_start_ms": origin.elapsed_time(slot["events"][p["start"]]),
                    "stream_end_ms": origin.elapsed_time(slot["events"][p["end"]]),
                    "host_enqueue_ms": (p["host_end_ns"] - p["host_start_ns"]) / 1e6,
                })
            self.records.append({
                "metadata": slot["metadata"], "status": slot["status"],
                "stream_step_ms": origin.elapsed_time(done),
                "host_step_ms": (slot["host_end_ns"] - slot["host_start_ns"]) / 1e6,
                "commit_lengths": slot["commits"][:slot["commit_count"]].tolist(),
                "phases": phases,
            })
            self.free.append(index)
        return len(self.pending)

    def wrap(self, obj, attr, phase_name):
        old = getattr(obj, attr)
        had_instance_attr = attr in vars(obj)
        previous_instance_attr = vars(obj).get(attr)

        @functools.wraps(old)
        def wrapped(*args, **kwargs):
            if self.active is None:
                return old(*args, **kwargs)
            with self.phase(phase_name):
                return old(*args, **kwargs)

        setattr(obj, attr, wrapped)
        self._undo.append((obj, attr, had_instance_attr, previous_instance_attr))

    def uninstall(self):
        if self.active is not None:
            raise RuntimeError("cannot uninstall in a running step")
        for obj, attr, existed, previous in reversed(self._undo):
            if existed:
                setattr(obj, attr, previous)
            else:
                delattr(obj, attr)
        self._undo.clear()


def install_dflash_recorder(worker, recorder):
    """Wrap real objects without modifying serving files. Call after warm-up.

    Phase nesting is intentional; do not sum parent and child durations.
    Target forward is distinguished by top-level metadata mode.
    """
    if recorder._undo:
        raise RuntimeError("recorder is already installed")
    targets = [
        (worker.draft_model_runner, "forward", "draft_forward"),
        (worker.target_worker, "forward_batch_generation", "target_forward"),
        (worker, "_wait_for_fused_hicache_load", "hicache_wait"),
        (worker, "_accept_block", "accept"),
        (worker, "_append_target_hidden_to_draft_kv_by_loc", "draft_kv_append"),
        (worker.draft_model, "project_target_hidden", "context_projection"),
        (worker, "_append_target_hidden_fused", "kv_materialize_fused"),
        (worker, "_append_target_hidden_sequential", "kv_materialize_sequential"),
    ]
    lp = worker._target_logits_processor
    if lp is not None:
        targets.extend([(lp, "deferred_argmax", "target_argmax"),
                        (lp, "finish_deferred_gather", "target_full_gather")])
    for obj, attr, _ in targets + [(worker, "forward_batch_generation", "step")]:
        if not callable(getattr(obj, attr, None)):
            raise TypeError("missing callable: " + attr)
    try:
        for obj, attr, name in targets:
            recorder.wrap(obj, attr, name)
    except BaseException:
        recorder.uninstall()
        raise
    old = worker.forward_batch_generation
    existed = "forward_batch_generation" in vars(worker)
    previous = vars(worker).get("forward_batch_generation")

    @functools.wraps(old)
    def run(batch, *args, **kwargs):
        if not recorder.begin({}):
            return old(batch, *args, **kwargs)
        lengths = getattr(batch, "extend_lens", None)
        metadata = {
            "mode": str(batch.forward_mode),
            "requests": len(batch.seq_lens),
            "extend_tokens": sum(lengths) if isinstance(lengths, list) else None,
            "mixed_extend": bool(getattr(batch, "is_extend_in_batch", False)),
            "graph_eligible": bool(getattr(batch, "can_run_decode_cuda_graph", False)),
        }
        recorder.active["metadata"].update(metadata)
        try:
            out = old(batch, *args, **kwargs)
        except BaseException:
            recorder.end(status="forward_error")
            raise
        if recorder.active is not None:
            recorder.active["metadata"]["target_graph_used"] = bool(
                getattr(out, "can_run_cuda_graph", False))
        recorder.end(getattr(out, "accept_lens", None))
        return out

    worker.forward_batch_generation = run
    recorder._undo.append((worker, "forward_batch_generation", existed, previous))
    return recorder
