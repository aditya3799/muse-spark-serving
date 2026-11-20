"""Experimental exact BF16 global-scale plus stock CUDA SiLU fusion."""
from pathlib import Path
import torch
from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH
from sglang.kernels.jit.utils import (
    cache_once, get_activation_cuda_cflags, is_arch_support_pdl, load_jit, make_cpp_args,
)

@cache_once
def scale_activation_module():
    args = make_cpp_args(torch.bfloat16, is_arch_support_pdl())
    return load_jit(
        'experimental_exact_scale_silu', *args,
        cuda_files=[str(Path(__file__).with_suffix('.cuh'))],
        extra_cuda_cflags=get_activation_cuda_cflags(),
        extra_include_paths=[str(KERNEL_PATH / 'csrc')],
        cuda_wrappers=[('run', f'ScaledActivationKernel<{args}>::run')],
    )

def scaled_silu_and_mul(input, output, local_ids, scales):
    if input.ndim != 2 or output.ndim != 2 or scales.ndim != 1:
        raise ValueError('Expected two matrices and scale vector')
    if any(t.dtype != torch.bfloat16 for t in (input, output, scales)):
        raise TypeError('Only BF16 arithmetic is supported')
    if local_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError('Expected integer route IDs')
    if not input.is_cuda or any(t.device != input.device for t in (output, local_ids, scales)):
        raise ValueError('All tensors must share a CUDA device')
    if not all(t.is_contiguous() for t in (input, output, local_ids, scales)):
        raise ValueError('All tensors must be contiguous')
    if input.shape[0] != output.shape[0] or input.shape[1] != 2 * output.shape[1]:
        raise ValueError('Invalid input/output shape')
    if local_ids.numel() != input.shape[0] or scales.numel() == 0:
        raise ValueError('Invalid route/scale shape')
    # Match the stock filtered activation wrapper's integer normalization.
    ids = local_ids.reshape(-1).to(torch.int32)
    scale_activation_module().run(input, output, ids, scales)
    return output