"""Column-parallel bf16 GEMM + all-gather for a replicated projection at decode M.

``out = x @ weight.T (+ b) (+ c)`` where ``x`` is identical on every TP rank
(e.g. an all-reduced tensor) and every rank needs the full result. Each rank
runs the bf16 GEMM dispatcher on its contiguous ``N / world`` row block of the
weight (reading ``1 / world`` of the bytes), then one kernel multicast-stores the
slice into every peer's CustomAllReduceV2 push workspace and Lamport-spins the
full rows together. Any TP world size; needs the plane's multicast mapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.communication.muse_spark_all_gather import (
    MAX_TOKENS,
    all_gather,
)

if TYPE_CHECKING:
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )


def reserve_projection_workspace(comm: Optional[CustomAllReduceV2], n: int) -> None:
    """Reserve shared push storage during model construction, before capture.

    Keep the all-reduce thresholds and counter/launch geometry unchanged.
    The gather strides over its output using the existing counter capacity.
    """
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    if (
        isinstance(comm, CustomAllReduceV2)
        and not comm.disabled
        and comm.has_multicast
        and comm.world_size in (4, 8)
        and n in (4096, 8192)
    ):
        comm.reserve_push_capacity(MAX_TOKENS * n * 2)


def gemm_local(
    comm: CustomAllReduceV2, x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """This rank's ``[M, N / world]`` column slice of ``x @ weight.T``."""
    from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch

    n_local = weight.shape[0] // comm.world_size
    w_slice = weight[comm.rank * n_local : (comm.rank + 1) * n_local]
    return bf16_gemm_dispatch(x.contiguous(), w_slice, None)


def gemm_ag(
    comm: CustomAllReduceV2,
    x: torch.Tensor,
    weight: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    c: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``x @ weight.T (+ b) (+ c)``; ``weight`` is the FULL replicated [N, K]."""
    local = gemm_local(comm, x, weight)
    if out is None:
        out = torch.empty(x.shape[0], weight.shape[0], dtype=x.dtype, device=x.device)
    return all_gather(comm, local, out, b, c)


def prefill_row_projection_supported(group, x, weight) -> bool:
    """A large replicated projection with equal contiguous token-row shards."""
    if group is None or group.world_size not in (2, 4, 8):
        return False
    if not x.is_cuda or x.ndim != 2 or weight.ndim != 2:
        return False
    if x.dtype != torch.bfloat16 or weight.dtype != x.dtype:
        return False
    if weight.device != x.device or not weight.is_contiguous():
        return False
    m, k = x.shape
    n = weight.shape[0]
    return (
        torch.cuda.get_device_capability(x.device) == (9, 0)
        and m >= 512
        and m % group.world_size == 0
        and min(k, n) >= 1024
        and k == weight.shape[1]
        and k % 128 == 0
        and n % 128 == 0
    )


def prefill_row_projection(group, x, weight):
    """Compute full output rows once, then gather in original token order.

    Input and weights must be replicated on the TP group. The complete K
    reduction and full output dimension remain local, with FP32 output before
    the single BF16 rounding. No persistent repacked weight or padding exists.
    """
    m_local = x.shape[0] // group.world_size
    start = group.rank_in_group * m_local
    return _project_and_gather_rows(group, x[start : start + m_local], weight)


def reduce_scatter_prefill_row_projection(group, partials, weight):
    """Reduce contiguous rank-owned rows before applying the replicated weight."""
    local = partials.new_empty((partials.shape[0] // group.world_size, partials.shape[1]))
    group.reduce_scatter_tensor(local, partials)
    return _project_and_gather_rows(group, local, weight)


def _project_and_gather_rows(group, rows, weight):
    local_fp32 = torch.mm(rows, weight.t(), out_dtype=torch.float32)
    local = local_fp32.to(rows.dtype)
    del local_fp32
    return group.all_gather(local, dim=0)


def small_projection_supported(group, x, weight) -> bool:
    """Hopper column shards through an already-reserved multicast workspace."""
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    if group is None or group.world_size not in (2, 4, 8):
        return False
    comm = group.ca_comm
    if (
        not isinstance(comm, CustomAllReduceV2)
        or comm.disabled
        or not comm.has_multicast
        or comm.world_size != group.world_size
    ):
        return False
    if not x.is_cuda or x.ndim != 2 or weight.ndim != 2:
        return False
    if x.dtype != torch.bfloat16 or weight.dtype != x.dtype:
        return False
    if weight.device != x.device or not weight.is_contiguous():
        return False
    m, k = x.shape
    n = weight.shape[0]
    return (
        torch.cuda.get_device_capability(x.device) == (9, 0)
        and 1 <= m < 512
        and min(k, n) >= 1024
        and k == weight.shape[1]
        and k % 128 == 0
        and n % (128 * group.world_size) == 0
        and m * n * 2 <= comm.max_push_size
        and comm.config.num_push_blocks >= 2
    )


def small_projection_local(comm, x, weight):
    """Complete K reductions for this rank's contiguous output columns."""
    n_local = weight.shape[0] // comm.world_size
    start = comm.rank * n_local
    local_fp32 = torch.mm(
        x, weight[start : start + n_local].t(), out_dtype=torch.float32
    )
    return local_fp32.to(x.dtype)


def small_projection(group, x, weight):
    """Gather complete BF16 columns without an intermediate layout copy."""
    comm = group.ca_comm
    local = small_projection_local(comm, x, weight)
    output = torch.empty(
        x.shape[0], weight.shape[0], dtype=x.dtype, device=x.device
    )
    return all_gather(comm, local, output)
