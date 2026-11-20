# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM Humming PR 107, lesj0610, by selecting probed scale lanes in the pinned reader.
# https://github.com/vllm-project/humming/pull/107
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import torch
import humming
from humming import ops
from humming.kernel.humming import HummingKernel
from humming.layer import HummingMethod

HEADER_SHA = '315d2c6b0bc965dd77485a15b522c96585b11de15e39a69108cdb3b3de02f75b'
OLD_SWAP = '        scalar_t *dq_bs_scalar_ptr = reinterpret_cast<scalar_t *>(dq_bs);\n\n        PRAGMA_UNROLL\n        for (uint32_t j = 0; j < 2; j++) {\n          scalar_t tmp = dq_bs_scalar_ptr[2 + 4 * j];\n          dq_bs_scalar_ptr[2 + 4 * j] = dq_bs_scalar_ptr[1 + 4 * j];\n          dq_bs_scalar_ptr[1 + 4 * j] = tmp;\n        }\n\n'
OLD_INDEX = 'bs_f16_ptr[i] = this->num2num2(bs_half_ptr[i]);'
NEW_INDEX = 'if constexpr (ElementBS::kBits == 8 && !kUseFusedE8m0Scale) {\n            auto *raw_scale = reinterpret_cast<scalar_t *>(dq_bs);\n            const uint32_t lane = j < 4 ? (j / 2) * 4 + i * 2 + j % 2 : j * 2 + i;\n            bs_f16_ptr[i] = this->num2num2(raw_scale[lane]);\n          } else {\n            bs_f16_ptr[i] = this->num2num2(bs_half_ptr[i]);\n          }'
@lru_cache(maxsize=1)
def adapted_source():
    headers = Path(humming.__file__).resolve().parent / "include/humming"
    kernel = (headers / "kernel/humming.cuh").read_text()
    main = (headers / "arith/mainloop_arith.cuh").read_text()
    if hashlib.sha256(main.encode()).hexdigest() != HEADER_SHA:
        return None
    assert main.count(OLD_SWAP) == main.count(OLD_INDEX) == 1
    main = main.replace(OLD_SWAP, "").replace(OLD_INDEX, NEW_INDEX)
    return kernel.replace("#include <humming/arith/mainloop_arith.cuh>", main)

class ScaleOrderKernel(HummingKernel):
    def prepare(self):
        anchor = "#include <humming/kernel/humming.cuh>"
        assert self.code.count(anchor) == 1 and adapted_source() is not None
        self.code=self.code.replace(anchor,adapted_source())
        super().prepare()

@lru_cache(maxsize=256)
def kernel_handle(context, meta_json, compute_json, tuning_json):
    config=json.loads(meta_json)
    config.pop("sublayer_name",None)
    config.update(json.loads(compute_json))
    config.update(json.loads(tuning_json))
    num_sms=config.pop("num_sms",0)
    kernel=ScaleOrderKernel(**config)
    return torch.tensor([kernel.kernel_id,num_sms],dtype=torch.int64,device="cpu")

def eligible(kwargs):
    x=kwargs["inputs"]
    c=kwargs["tuning_config"]
    compute=kwargs["compute_config"]
    return (kwargs["valid_shape_m"]>=1 and kwargs["top_k"] in (1,8)
            and x.is_cuda and x.dtype==torch.bfloat16
            and isinstance(c,dict) and c.get("use_stream_k") is False
            and c.get("use_f16_accum") is False and not c.get("use_warp_spec")
            and not c.get("use_tma") and c.get("multi_cast_size_a",1)==1
            and c.get("multi_cast_size_b",1)==1
            and compute.get("gemm_type")=="indexed" and compute.get("use_f16_accum") is False)

def scale_order_forward(layer, **kwargs):
    if not eligible(kwargs) or adapted_source() is None:
        return HummingMethod.forward_layer(layer,**kwargs)
    meta=layer.humming_metas[kwargs["sublayer_name"]]
    if (meta.input_scale_group_size or meta.has_bias or meta.has_zero_point
            or str(meta.b_dtype)!="float4e2m1" or str(meta.bs_dtype)!="float8e4m3"
            or meta.weight_scale_group_size!=16):
        return HummingMethod.forward_layer(layer,**kwargs)
    handles=kernel_handle(HummingKernel.current_context(),meta.to_str(),
        json.dumps(kwargs["compute_config"],sort_keys=True),json.dumps(kwargs["tuning_config"],sort_keys=True))
    return ops.launch_kernel(
        configs=handles,inputs=kwargs["inputs"],outputs=kwargs["outputs"],
        weight=getattr(layer,meta.weight_name),weight_scale=getattr(layer,meta.weight_scale_name),
        weight_scale_2=getattr(layer,meta.weight_scale_2_name,None),
        sorted_ids=kwargs["sorted_ids"],expert_ids=kwargs["expert_ids"],
        num_tokens_padded=kwargs["num_tokens_padded"],locks=getattr(layer,"locks",None),
        top_k=kwargs["top_k"],valid_shape_m=kwargs["valid_shape_m"])
