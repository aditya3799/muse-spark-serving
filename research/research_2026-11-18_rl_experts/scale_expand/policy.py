"""Build an alternate canonical group-scale layout without duplicating 4-bit codes."""
import copy
import dataclasses
import torch
from humming import dtypes


def expanded(layer, stage):
    result=copy.copy(layer)
    result._parameters=dict(layer._parameters)
    result._buffers=dict(layer._buffers)
    result.humming_metas=dict(layer.humming_metas)
    old=layer.humming_metas[stage]
    assert str(old.bs_dtype)=="float8e4m3" and not old.use_int_weight_scale and old.weight_scale_group_size==16
    assert not old.should_apply_bs_on_c and not old.use_fused_e8m0_scale
    new=dataclasses.replace(old,bs_dtype=dtypes.bfloat16)
    result.humming_metas[stage]=new
    weight=getattr(layer,old.weight_name)
    scales=getattr(layer,old.weight_scale_name)
    assert scales.dtype==torch.float8_e4m3fn
    value=scales.to(dtype=torch.bfloat16)
    setattr(result,old.weight_scale_name,torch.nn.Parameter(value,requires_grad=False))
    assert weight.data_ptr()==getattr(result,new.weight_name).data_ptr()
    assert getattr(layer,old.weight_scale_name).dtype==torch.float8_e4m3fn
    return result
