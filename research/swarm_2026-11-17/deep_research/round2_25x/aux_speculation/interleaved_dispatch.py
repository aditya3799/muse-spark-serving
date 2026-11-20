"""Research-only single-copy W13 activation dispatch, callbacks inject native APIs."""
def activate(layout, x, out, ids, canonical, interleaved):
    import torch
    if layout not in ('canonical','gate_up_interleaved'):raise ValueError('Unknown W13 layout')
    if x.dtype!=torch.bfloat16 or out.dtype!=x.dtype:raise ValueError('Prepared BF16 layer required')
    if x.ndim!=2 or out.ndim!=2 or x.shape[0]!=out.shape[0] or x.shape[1]!=2*out.shape[1]:raise ValueError('Activation geometry')
    if out.shape[1]%8:raise ValueError('Native BF16 vector width')
    if ids.dtype!=torch.int32 or ids.ndim!=1 or ids.numel()!=x.shape[0]:raise ValueError('One int32 expert ID per row')
    if any(not t.is_contiguous() for t in (x,out,ids)):raise ValueError('Contiguous buffers required')
    if x.device!=out.device or x.device!=ids.device:raise ValueError('Device mismatch')
    def overlaps(a,b):
        return bool(a.numel() and b.numel() and a.data_ptr()<b.data_ptr()+b.numel()*b.element_size() and b.data_ptr()<a.data_ptr()+a.numel()*a.element_size())
    if overlaps(out,x) or overlaps(out,ids):raise ValueError('Out-of-place required')
    return (canonical if layout=='canonical' else interleaved)(x,out,ids)
