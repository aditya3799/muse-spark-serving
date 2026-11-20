"""Isolated research support for fixed draft / shorter target verification.

Not wired to a server flag. A fixture must explicitly configure the worker.
This precursor uses eager target verification for non-default widths. Target
shape-dependent numerical behavior and end-to-end serving remain unqualified.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrefixVerificationConfig:
    draft_width: int
    verify_width: int

    def __post_init__(self):
        if not 1 <= self.verify_width <= self.draft_width:
            raise ValueError("verify width must be in [1, draft width]")


def configure_prefix_verification(worker, verify_width):
    # Configuration is host-side and must be identical on every TP/EP rank.
    # No adaptive policy or rank-local measurement is installed here.
    worker._prefix_verification_config = PrefixVerificationConfig(
        int(worker.block_size), int(verify_width)
    )


def resolve_prefix_verification_width(worker, batch, parallel):
    default = int(worker.block_size)
    config = getattr(worker, "_prefix_verification_config", None)
    if config is None or config.verify_width == default:
        return default
    if config.draft_width != default:
        raise RuntimeError("draft width changed after prefix configuration")
    runner = worker.target_worker.model_runner
    backend = runner.attn_backend
    supported = (
        type(backend).__name__ == "FlashAttentionBackend"
        and backend.fa_impl_ver == 3
        and backend.topk <= 1
        and not worker._need_mamba_verify_commit
        and not parallel.attn_dp_enabled
        and parallel.attn_cp_size == 1
        and parallel.attn_dcp_size == 1
        and parallel.pp_size == 1
        and not runner.model_config.is_encoder_decoder
        and runner.lora_manager is None
        and worker.lilicorr is None
        and not worker._is_domino
        and worker._can_skip_verify_logits(batch, batch.sampling_info)
    )
    return config.verify_width if supported else default


def next_prefix_accept_buffers(worker, batch_size, width):
    """Same dtypes and two-slot consumer lifetime as stock accept buffers."""
    states = getattr(worker, "_prefix_accept_buffers", None)
    if states is None:
        states = worker._prefix_accept_buffers = {}
    state = states.get(width)
    if state is None or state["capacity"] < batch_size:
        capacity = max(batch_size, 0 if state is None else 2 * state["capacity"])
        device = worker.device
        state = {
            "capacity": capacity,
            "slot": 0,
            "accept": torch.empty(capacity, dtype=torch.int32, device=device),
            "commit": [torch.empty(capacity, dtype=torch.int32, device=device) for _ in range(2)],
            "bonus": [torch.empty(capacity, dtype=torch.int64, device=device) for _ in range(2)],
            "tokens": [torch.empty((capacity, width), dtype=torch.int64, device=device) for _ in range(2)],
            "length": [torch.empty(capacity, dtype=torch.int64, device=device) for _ in range(2)],
        }
        states[width] = state
    slot = state["slot"]
    state["slot"] = (slot + 1) % 2
    return (
        state["accept"][:batch_size],
        state["commit"][slot][:batch_size],
        state["bonus"][slot][:batch_size],
        state["tokens"][slot][:batch_size],
        state["length"][slot][:batch_size],
    )
