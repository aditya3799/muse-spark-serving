"""Research-only lossless KV staging. No serving hook or default activation.

The arena must be reserved outside capture and retained for the lifetime of every
graph that references it. One arena belongs to one CUDA stream. FP8 bytes are
copied as uint8, with no conversion or scaling. Descale tensors stay at the caller.
"""
import threading

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["CAP", "TABLE_STRIDE", "CACHE_ROWS"])
def _pack(K, V, Table, Lengths, OutK, OutV,
          CAP, WIDTH: tl.constexpr, PAGE: tl.constexpr,
          TABLE_STRIDE, CACHE_ROWS,
          BLOCK: tl.constexpr):
    batch = tl.program_id(1)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // WIDTH
    col = offsets % WIDTH
    length = tl.load(Lengths + batch)
    valid = (row < CAP) & (row < length)
    page = tl.load(Table + batch * TABLE_STRIDE + row // PAGE,
                   mask=valid, other=0).to(tl.int64)
    source = (page * PAGE + row % PAGE) * WIDTH + col
    # Valid physical IDs are the same precondition required by native attention.
    # Invalid entries beyond a sequence's length are never dereferenced.
    kval = tl.load(K + source, mask=valid, other=0)
    vval = tl.load(V + source, mask=valid, other=0)
    target = batch.to(tl.int64) * CAP * WIDTH + offsets
    tl.store(OutK + target, kval, mask=row < CAP)
    tl.store(OutV + target, vval, mask=row < CAP)


class Arena:
    def __init__(self, *, budget=128 * 1024 * 1024):
        self.budget = int(budget)
        self.storage = None
        self.geometry = None
        self.owner = None

    def reserve(self, key, value, table, lengths):
        if self.storage is not None:
            raise RuntimeError("arena storage is immutable; allocate a separate owner")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("reserve must precede capture")
        if key.shape != value.shape or key.dtype != value.dtype:
            raise ValueError("identical KV storage geometry required")
        if key.ndim != 4 or not key.is_contiguous() or not value.is_contiguous():
            raise ValueError("contiguous [pages,page,heads,dim] required")
        if key.dtype not in (torch.float8_e4m3fn, torch.bfloat16):
            raise ValueError("only FP8 storage and BF16 diagnostic accepted")
        if table.ndim != 2 or table.stride(1) != 1 or table.dtype != torch.int32:
            raise ValueError("native int32 page table required")
        if lengths.shape != (table.shape[0],) or lengths.dtype != torch.int32:
            raise ValueError("native batch lengths required")
        if not all(t.device == key.device for t in (value, table, lengths)):
            raise ValueError("same device required")
        batch, pages = table.shape
        _, page, heads, dim = key.shape
        capacity = pages * page
        count = batch * capacity * heads * dim
        size = 2 * count * key.element_size()
        if size > self.budget:
            raise ValueError("bounded arena exceeded")
        self.geometry = (tuple(key.shape), key.dtype, table.shape,
                         table.stride(0), key.device, capacity, heads, dim)
        self.owner = (threading.get_ident(), torch.cuda.current_stream().cuda_stream)
        self.storage = torch.empty((2, batch, capacity, heads, dim),
                                   dtype=key.dtype, device=key.device)
        return size

    def __call__(self, key, value, table, lengths):
        shape, dtype, tshape, stride, device, cap, heads, dim = self.geometry
        if (tuple(key.shape), key.dtype, table.shape, table.stride(0), key.device) != (
                shape, dtype, tshape, stride, device):
            raise ValueError("geometry change requires a separate arena")
        if (threading.get_ident(), torch.cuda.current_stream().cuda_stream) != self.owner:
            raise RuntimeError("arena has one stream owner")
        if value.shape != key.shape or value.dtype != dtype:
            raise ValueError("KV geometry mismatch")
        if not key.is_contiguous() or not value.is_contiguous():
            raise ValueError("KV storage must stay contiguous")
        if lengths.shape != (tshape[0],) or lengths.dtype != torch.int32:
            raise ValueError("length geometry changed")
        # uint8 view also copies BF16 diagnostic bytes without arithmetic.
        width = heads * dim * key.element_size()
        _pack[(triton.cdiv(cap * width, 1024), table.shape[0])](
            key.view(torch.uint8), value.view(torch.uint8), table, lengths,
            self.storage[0].view(torch.uint8), self.storage[1].view(torch.uint8),
            cap, width, shape[1], stride, shape[0] * shape[1], 1024)
        return self.storage[0], self.storage[1]


def native_contiguous(flash, q, key, value, table, lengths, arena, **kwargs):
    """Caller supplies unchanged native arguments, especially split/scales/cuQ.

    PackGQA=True is required to preserve the page-size-1 native choice. This
    helper is only a diagnostic: it does not certify Hopper FP8 output parity.
    """
    packed_k, packed_v = arena(key, value, table, lengths)
    return flash(q, packed_k, packed_v, cache_seqlens=lengths,
                 page_table=None, pack_gqa=True, **kwargs)
