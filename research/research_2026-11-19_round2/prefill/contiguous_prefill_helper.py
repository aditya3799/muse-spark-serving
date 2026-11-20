"""Unqualified diagnostic only. Disabled unless explicitly enabled at startup.

No allocation/import of the staging kernel occurs while disabled. Enabling is
not an assertion of Hopper numerical parity: the full native FP8 gate is pending.
"""
import logging
import os
import threading

logger = logging.getLogger(__name__)
ENABLED = os.environ.get("SGLANG_EXPERIMENTAL_CONTIGUOUS_PREFILL_KV", "0") == "1"
MAX_BYTES = 128 * 1024 * 1024


def is_hopper(torch, device):
    return torch.cuda.get_device_capability(device)[0] == 9


def lifecycle_allowed(*, enabled, allowed, capture_setup, full_graph, capturing):
    return bool(enabled and allowed and not (capture_setup or full_graph or capturing))


def maybe_stage(backend, q, key, value, table, lengths, layer, forward_batch, *, allowed):
    # Keep this first: disabled mode never imports CUDA code or allocates memory.
    if not ENABLED or not allowed:
        return None
    import torch
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner_utils.prefill_graph import is_in_full_prefill_graph
    from sglang.srt.runtime_context import get_flags, get_parallel

    if not lifecycle_allowed(
        enabled=True, allowed=forward_batch.forward_mode == ForwardMode.EXTEND,
        capture_setup=get_flags().dp.capturing_prefill_graph,
        full_graph=is_in_full_prefill_graph(), capturing=torch.cuda.is_current_stream_capturing(),
    ):
        return None
    parallel = get_parallel()
    if parallel.pp_size != 1 or parallel.enable_prefill_cp:
        return None
    if (backend.fa_impl_ver != 3 or backend.page_size != 1 or backend.kv_cache_is_mxfp8
            or backend.use_mla or backend.fa_skip_kv_cache
            or layer.is_cross_attention or layer.head_dim != 64 or layer.v_head_dim != 64
            or layer.tp_k_head_num != layer.tp_v_head_num
            or layer.tp_q_head_num % layer.tp_k_head_num
            or forward_batch.dcp_kv_mask is not None):
        return None
    if (q.dtype != torch.float8_e4m3fn or key.dtype != q.dtype or value.dtype != q.dtype
            or key.ndim != 4 or key.shape != value.shape or key.shape[1] != 1
            or tuple(key.shape[-2:]) != (layer.tp_k_head_num, 64)
            or not key.is_contiguous() or not value.is_contiguous()
            or table.ndim != 2 or table.dtype != torch.int32 or table.stride(1) != 1
            or lengths.dtype != torch.int32 or lengths.shape != (table.shape[0],)
            or not all(t.device == key.device for t in (q, value, table, lengths))):
        return None
    count = table.shape[0] * table.shape[1] * layer.tp_k_head_num * 64
    if count <= 0 or 2 * count > MAX_BYTES:
        return None
    if not is_hopper(torch, key.device):
        return None
    state = getattr(backend, "_contiguous_prefill_kv_state", None)
    if state is False:
        return None
    owner = (threading.get_ident(), torch.cuda.current_stream(key.device).cuda_stream)
    if state is None:
        from sglang.kernels.ops.kvcache.contiguous_kv_staging import _pack
        before = torch.cuda.memory_allocated(key.device)
        reserved_before = torch.cuda.memory_reserved(key.device)
        try:
            storage = torch.empty((2, MAX_BYTES // 2), dtype=torch.uint8, device=key.device)
        except torch.cuda.OutOfMemoryError:
            backend._contiguous_prefill_kv_state = False
            logger.warning("CONTIGUOUS_PREFILL_KV disabled: bounded arena allocation failed")
            return None
        state = (owner, storage, _pack)
        backend._contiguous_prefill_kv_state = state
        logger.warning(
            "CONTIGUOUS_PREFILL_KV unqualified diagnostic enabled: arena=%d "
            "allocated_before=%d allocated_after=%d reserved_before=%d reserved_after=%d",
            MAX_BYTES, before, torch.cuda.memory_allocated(key.device),
            reserved_before, torch.cuda.memory_reserved(key.device),
        )
    if state[0] != owner or state[1].device != key.device:
        return None
    storage, pack = state[1:]
    batch, capacity = table.shape
    width = layer.tp_k_head_num * 64
    # Views alias ONE process-lifetime allocation across all full-attention layers.
    packed_k = storage[0, :count].view(batch, capacity, layer.tp_k_head_num, 64).view(q.dtype)
    packed_v = storage[1, :count].view_as(packed_k.view(torch.uint8)).view(q.dtype)
    pack[((capacity * width + 1023) // 1024, batch)](
        key.view(torch.uint8), value.view(torch.uint8), table, lengths,
        packed_k.view(torch.uint8), packed_v.view(torch.uint8),
        capacity, width, 1, table.stride(0), key.shape[0], 1024,
    )
    # CUDA errors from invalid live IDs or a failed kernel are fatal, matching
    # native attention. Do not hide an asynchronous failure with a fallback.
    return packed_k, packed_v
