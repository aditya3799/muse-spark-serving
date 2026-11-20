"""Qualify the loaded native DFlash projector on the scheduler's startup forward stream."""
import logging

import torch

from . import CompleteProjection
from .backend import describe, native_workspace

logger = logging.getLogger(__name__)


@torch.inference_mode()
def prewarm_projector(model, batch_sizes, block_size, group):
    fc = getattr(model, "fc", None)
    if (getattr(model, "quantizable", True) or type(fc) is not torch.nn.Linear
        or fc.bias is not None or not fc.weight.is_cuda or fc.weight.dtype != torch.bfloat16):
        return
    if getattr(model, "_context_projector", None) is not None:
        return
    runner = CompleteProjection(fc, empirical=True)
    local, keys = {}, {}
    w = fc.weight
    generator = torch.Generator(device=w.device)
    budget = native_workspace()
    try:
        for bs in sorted(set(batch_sizes or ())):
            rows = int(bs) * int(block_size)
            if budget is None or rows < 128:
                continue
            # Cold tensors are temporary. Bound each trial by both available memory
            # and the active native backend's ordinary bounded scratch reservation.
            free, _ = torch.cuda.mem_get_info(w.device)
            limit = min(free // 8, 16 * budget[1])
            estimate = rows * (2 * fc.in_features + 6 * fc.out_features) * w.element_size() + budget[1]
            if estimate > limit:
                continue
            trials = [
                torch.empty((rows, fc.in_features), dtype=w.dtype, device=w.device).normal_(generator=generator)
                for _ in range(2)
            ]
            info = runner.prepare(trials[:1], trials[1:])
            if info.get("admitted"):
                key = runner.key(trials[0])
                (handle, index, _), _ = runner.entries[key]
                fields = describe(runner.lib, handle, index)
                local[rows] = tuple(fields[n] for n in (
                    "id","tile","split","reduction","swizzle","custom","stages","inner","cluster"))
                keys[rows] = key
            del trials
    except (OSError, RuntimeError, ValueError) as error:
        logger.warning("DFlash native projector tuning stopped during startup: %s", error)
    # Every participating rank keeps only plans it has separately qualified and
    # whose reduction/tiling signatures agree across the ordinary draft TP group.
    peers = group.all_gather_object(local) if group.world_size > 1 else [local]
    for rows, key in keys.items():
        if any(peer.get(rows) != local[rows] for peer in peers):
            (handle, _, _), _ = runner.entries.pop(key)
            runner.lib.drop_sweep(handle)
    if runner.entries:
        model._context_projector = runner
        logger.info("DFlash qualified %d full-width native projection geometries.", len(runner.entries))

