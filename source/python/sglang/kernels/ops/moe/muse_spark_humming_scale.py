"""Exact BF16 global scaling, skipping non-local expert routes."""
import torch
import triton
import triton.language as tl


@triton.jit
def _scale_rows_kernel(x_ptr, ids_ptr, scale_ptr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    expert = tl.load(ids_ptr + row)
    if expert >= 0:
        offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(x_ptr + row * WIDTH + offset, mask=offset < WIDTH, other=0).to(tl.float32)
        scale = tl.load(scale_ptr + expert).to(tl.float32)
        # Preserve the separate BF16 multiplication after the BF16 GEMM output.
        result = (value * scale).to(x_ptr.dtype.element_ty)
        tl.store(x_ptr + row * WIDTH + offset, result, mask=offset < WIDTH)


def scale_active_rows(output, local_ids, scales):
    if output.ndim != 2 or scales.ndim != 1:
        raise ValueError("Expected a matrix output and one scale per expert")
    if output.dtype != torch.bfloat16 or scales.dtype != torch.bfloat16:
        raise TypeError("Global scaling requires BF16 output and scales")
    if local_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("Local expert IDs must be integer tensors")
    if not output.is_cuda or not output.device == local_ids.device == scales.device:
        raise ValueError("Output, IDs and scales must share one CUDA device")
    if not all(t.is_contiguous() for t in (output, local_ids, scales)):
        raise ValueError("Output, IDs and scales must be contiguous")
    if output.shape[0] != local_ids.numel() or scales.numel() == 0:
        raise ValueError("Expected one local ID per output row and nonempty scales")
    width = output.shape[1]
    if width == 0:
        raise ValueError("Output width must be positive")
    if output.shape[0] == 0:
        return
    block = min(8192, triton.next_power_of_2(width))
    _scale_rows_kernel[(output.shape[0], triton.cdiv(width, block))](
        output, local_ids, scales, WIDTH=width, BLOCK=block,
        num_warps=8 if block >= 8192 else 4,
    )
