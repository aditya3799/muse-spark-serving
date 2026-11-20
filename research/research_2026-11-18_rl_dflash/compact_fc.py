"""Projection trial on the existing DFlash2 BF16 context projection."""

import torch


def compact_fc(hidden, weight, lengths, width):
    """Project only live tokens. The caller consumes only their output rows.

    This eager path synchronizes once to obtain the nonzero row count. A
    capture never enters it. The full-acceptance path retains the original
    native BF16 matmul geometry. Both arms omit the same downstream RMSNorm.
    """
    if hidden.is_cuda and torch.cuda.is_current_stream_capturing():
        return torch.nn.functional.linear(hidden, weight)
    live = (torch.arange(width, device=hidden.device) < lengths[:, None]).reshape(-1)
    indices = live.nonzero().flatten()
    if indices.numel() == hidden.shape[0]:
        return torch.nn.functional.linear(hidden, weight)
    compact = torch.index_select(hidden, 0, indices)
    projected = torch.nn.functional.linear(compact, weight)
    expanded = projected.new_zeros((hidden.shape[0], weight.shape[0]))
    return expanded.index_copy_(0, indices, projected)
