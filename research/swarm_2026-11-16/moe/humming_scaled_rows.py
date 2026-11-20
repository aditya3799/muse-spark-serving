"""Experimental exact BF16 global scaling, skipping non-local routes."""
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
    assert output.dtype == scales.dtype == torch.bfloat16
    assert output.is_contiguous() and local_ids.is_contiguous()
    assert output.shape[0] == local_ids.numel()
    width = output.shape[1]
    block = min(8192, triton.next_power_of_2(width))
    _scale_rows_kernel[(output.shape[0], triton.cdiv(width, block))](
        output, local_ids, scales, WIDTH=width, BLOCK=block,
        num_warps=8 if block >= 8192 else 4,
    )
