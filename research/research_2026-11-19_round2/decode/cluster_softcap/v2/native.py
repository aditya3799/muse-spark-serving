import triton
import triton.language as tl
from triton.language.extra import libdevice
@triton.jit
def _softcap_row_argmax_kernel(
    logits_ptr,
    out_ptr,
    row_stride,
    valid_width,
    col_offset,
    softcapping_value,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    best_val = -float("inf")
    best_idx = 0
    for start in range(0, valid_width, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < valid_width
        x = tl.load(logits_ptr + row * row_stride + cols, mask=mask, other=0.0)
        x = x / softcapping_value
        x = libdevice.tanh(x)
        x = x * softcapping_value
        x = tl.where(mask, x, -float("inf"))
        block_val, block_idx = tl.max(
            x, axis=0, return_indices=True, return_indices_tie_break_left=True
        )
        take = block_val > best_val
        best_idx = tl.where(take, start + block_idx, best_idx)
        best_val = tl.where(take, block_val, best_val)
    tl.store(out_ptr + row * 2, best_val)
    tl.store(out_ptr + row * 2 + 1, (best_idx + col_offset).to(tl.float32))
