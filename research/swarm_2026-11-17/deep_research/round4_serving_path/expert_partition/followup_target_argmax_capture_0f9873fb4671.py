"""Research-only target argmax capture tail. No installation into SGLang.

Retains the exact existing logits_processor.deferred_argmax implementation.
Eligibility is intentionally narrow: serial target execution, no graph variants,
DP/CP/PP, LoRA or PDMux, exact target-verify bucket, fixed width. The caller must
register before target capture and pass the actual replay runner to consume().
All ranks must install the hook identically. Real NCCL validation is pending.
"""
import torch


def fingerprint(t):
    return (t.data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()),
            t.dtype, t.device, t.untyped_storage().nbytes())


class TargetArgmaxCapture:
    def __init__(self, logits_processor, *, max_rows, width, device):
        if max_rows < 1 or width < 1 or max_rows % width:
            raise ValueError('invalid fixed-width output capacity')
        self.lp = logits_processor
        self.width = width
        self.output = torch.empty(max_rows, dtype=torch.long, device=device)
        self._fingerprint = fingerprint(self.output)
        self._captures = set()
        self._pending = set()
        self._runner_refs = {}
        self.generation = 0

    def invalidate(self):
        # Must be called before runner replacement/recapture or topology change.
        self._captures.clear()
        self._pending.clear()
        self._runner_refs.clear()
        self.generation += 1

    def capture_hook(self, runner, out, forward_batch, num_tokens):
        if (not forward_batch.forward_mode.is_target_verify()
                or runner.captured_req_width != self.width
                or num_tokens < 1 or num_tokens > self.output.numel()
                or num_tokens % self.width
                or fingerprint(self.output) != self._fingerprint):
            return
        logits = out.next_token_logits
        if logits is None or logits.shape[0] != num_tokens or not self.lp.is_deferred_shard(logits):
            return
        # Always runs in this graph. Non-greedy callers ignore the side output
        # and retain their original full-gather path, but pay this extra work.
        ids = self.lp.deferred_argmax(logits)
        self.output[:num_tokens].copy_(ids)
        if torch.cuda.is_current_stream_capturing():
            self._runner_refs[id(runner)] = runner
            self._pending.add((id(runner), num_tokens, self.width, self.generation))

    def confirm_capture(self, *, runner, captured_rows, generation):
        # Call only after the backend has successfully installed this graph key.
        # Warmup and a failed capture must never authorize consumption.
        key = (id(runner), captured_rows, self.width, generation)
        if (generation != self.generation or key not in self._pending
                or self._runner_refs.get(id(runner)) is not runner):
            return False
        self._pending.remove(key)
        self._captures.add(key)
        return True

    def consume(self, *, runner, raw_rows, captured_rows, width,
                graph_used, can_skip_verify_logits, generation):
        # Exact-bucket admission avoids silently consuming padded/stale rows.
        if (not graph_used or not can_skip_verify_logits
                or self._runner_refs.get(id(runner)) is not runner
                or raw_rows != captured_rows or width != self.width
                or generation != self.generation
                or fingerprint(self.output) != self._fingerprint
                or (id(runner), captured_rows, width, generation) not in self._captures):
            return None
        return self.output[:raw_rows]
