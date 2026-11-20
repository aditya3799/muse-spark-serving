from pathlib import Path

import torch

from sglang.kernels.jit.utils import cache_once, get_activation_cuda_cflags, is_arch_support_pdl, load_jit, make_cpp_args
from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH


@cache_once
def route_activation_module():
    args = make_cpp_args(torch.bfloat16, is_arch_support_pdl())
    return load_jit(
        "muse_route_scaled_activation", *args,
        cuda_files=["elementwise/route_activation.cuh"],
        extra_cuda_cflags=get_activation_cuda_cflags(),
        extra_include_paths=[str(KERNEL_PATH / "csrc")],
        cuda_wrappers=[("run", f"RouteActivationKernel<{args}, 8>::run")],
    )


def scaled_prefill_activation(x, out, ids, scales):
    if (
        ids.ndim == 2 and ids.shape[0] >= 512 and ids.shape[1] == 8
        and x.ndim == 2 and out.ndim == 2 and x.shape == (ids.numel(), 8192)
        and out.shape == (ids.numel(), 4096) and scales.ndim == 1 and scales.numel() > 0
        and x.is_cuda and x.device == out.device == ids.device == scales.device
        and x.dtype == out.dtype == scales.dtype == torch.bfloat16 and ids.dtype == torch.int32
        and x.is_contiguous() and out.is_contiguous() and ids.is_contiguous() and scales.is_contiguous()
        and ids.numel() <= 2**32 // 1024 - 1
    ):
        route_activation_module().run(x, out, ids.view(-1), scales)
    else:
        from sglang.kernels.ops.activation.activation import silu_and_mul
        from sglang.kernels.ops.moe.muse_spark_humming_scale import scale_active_rows
        scale_active_rows(x, ids, scales)
        silu_and_mul(x, out, expert_ids=ids.flatten(), expert_step=1)
