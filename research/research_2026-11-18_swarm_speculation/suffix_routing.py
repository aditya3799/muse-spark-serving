"""Proposal-side adapter for a greedy DFlash verifier with an 8-token draft block.

The caller owns request admission, per-row eligibility, coherent graph dispatch
and the target verifier. All decode operations accept device tensors. The
store and producer must share a stream or order operations with CUDA events.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CopyRoutes:
    # The proposal includes the existing bonus anchor in column zero.
    proposals: torch.Tensor
    copy_rows: torch.Tensor
    verify_widths: torch.Tensor
    all_copy: torch.Tensor
    uniform_width: torch.Tensor


class DFlashCopyRouter:
    def __init__(self, store, *, stock_width=8, copy_width=32):
        if not 1 < stock_width < copy_width <= store.capacity:
            raise ValueError("invalid draft or verify width")
        self.store = store
        self.stock_width = stock_width
        self.copy_width = copy_width

    def finish_prefill(self, slot, request_id, committed_prompt, target_bonus):
        """Call once after the final chunk, using the full actual prefix.

        committed_prompt includes any radix hit and prior resumed generation.
        target_bonus is the new target token, already emitted by the prefill.
        """
        prefix = torch.as_tensor(committed_prompt, device=self.store.device)
        bonus = torch.as_tensor(target_bonus, device=self.store.device)
        if bonus.numel() != 1 or prefix.ndim != 1:
            raise ValueError("need a single target bonus and a flat prompt")
        if prefix.numel() == 0:
            prefix = prefix.to(dtype=bonus.dtype)
        confirmed = torch.cat((prefix, bonus.reshape(1)))
        return self.store.bind(slot, request_id, confirmed)

    def route(self, slots, epochs, kv_prefix_lens, bonus_tokens, remaining, eligible):
        """eligible is a bool vector with request and backend constraints applied.

        DFlash's prefix_lens counts materialized KV. The bonus belongs to the
        confirmed suffix history and occupies the first row of the next verify.
        """
        kv = self.store._ids(kv_prefix_lens)
        eligible = torch.as_tensor(eligible, device=self.store.device)
        if eligible.dtype != torch.bool or eligible.shape != kv.shape:
            raise ValueError("eligible must be an aligned boolean vector")
        proposal, found = self.store.propose(
            slots, epochs, kv + 1, bonus_tokens, remaining, width=self.copy_width
        )
        use = found & eligible
        widths = torch.where(use, self.copy_width, self.stock_width)
        # A fully idle batch stays on SGLang's native idle path.
        all_copy = use.all() & (use.numel() > 0)
        uniform = torch.where(all_copy, self.copy_width, self.stock_width)
        return CopyRoutes(proposal, use, widths, all_copy, uniform)

    def accept(self, slots, epochs, kv_prefix_lens, emitted, commit_lens):
        """Append newly emitted tokens (accepted drafts and next target bonus)."""
        return self.store.append_verified(
            slots, epochs, self.store._ids(kv_prefix_lens) + 1, emitted, commit_lens
        )
