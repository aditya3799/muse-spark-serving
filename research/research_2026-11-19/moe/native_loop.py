# SPDX-License-Identifier: Apache-2.0
# Research backport of fixed-cycle mainloop and live-prefetch bounds from Humming.
# https://github.com/vllm-project/humming/blob/0832818a89e2de2b3b672ac1ebd076ba18e0a994/humming/include/humming/kernel/humming.cuh
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import torch
import humming
from humming import ops
from humming.kernel.humming import HummingKernel
from humming.layer import HummingMethod

HEADER_SHA = '9a021dffc15f2ba2f3c355b4e508aeb3722b351a8be8130ffd4530743df3edd3'
VARIANT = 'cycles'
@lru_cache(maxsize=8)
def source_for(variant):
    headers = Path(humming.__file__).resolve().parent / 'include/humming'
    kernel = (headers / 'kernel/humming.cuh').read_text()
    if hashlib.sha256(kernel.encode()).hexdigest() != HEADER_SHA:
        return None
    start = kernel.index('    while (slice_iters) {')
    end = kernel.index('\n    consumer.wait_channel();',start)
    text = kernel[start:end]
    if variant in ('tail','cycles_tail'):
        needle='          s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1);'
        assert text.count(needle)==1
        text=text.replace(needle,'          if (slice_iters > 1 || warp_iter_id + 1 < Ctx::kWarpIters)\n'+needle)
        needle='          mma.transform_b(\n              (warp_iter_id + 1) % 2,\n              (warp_iter_id + 1) % Ctx::kWarpIters);'
        assert text.count(needle)==1
        text=text.replace(needle,'          if (slice_iters > 1 || warp_iter_id + 1 < Ctx::kWarpIters)\n'+needle)
    if variant in ('cycles','cycles_tail','countdown'):
        l=text.index('        debug_kernel_timeout_check')
        r=text.index('        slice_iters--;')
        stage=text[l:r].replace('slice_iters','remaining_iters')
        text=('    static_assert(!Ctx::kUseStreamK);\n'
              '    constexpr uint32_t kSliceIters = ProblemShape::K / BlockShape::K;\n'
              '    auto consume_stage = [&](uint32_t stage_id, uint32_t remaining_iters) {\n'+stage+
              '    };\n'
              '    #pragma unroll 1\n'
              '    for (uint32_t cycle = 0; cycle < kSliceIters / kNumStages; ++cycle) {\n'
              '      PRAGMA_UNROLL\n'
              '      for (uint32_t stage = 0; stage < kNumStages; ++stage)\n'
              '        consume_stage(stage, kSliceIters - cycle*kNumStages - stage);\n'
              '    }\n'
              '    PRAGMA_UNROLL\n'
              '    for (uint32_t stage = 0; stage < kSliceIters % kNumStages; ++stage)\n'
              '      consume_stage(stage, kSliceIters % kNumStages - stage);\n')
    if variant == 'countdown':
        text = text.replace('for (uint32_t cycle = 0; cycle < kSliceIters / kNumStages; ++cycle)',
                            'for (uint32_t remaining = kSliceIters; remaining >= kNumStages; remaining -= kNumStages)')
        text = text.replace('kSliceIters - cycle*kNumStages - stage','remaining - stage')
    if variant not in ('cycles','tail','cycles_tail','countdown'):
        raise ValueError(variant)
    text='    static_assert(Ctx::kWarpIters % 2 == 0, "requires alternating operand register banks");\n'+text
    return kernel[:start]+text+kernel[end:]

def adapted_source():
    return source_for(VARIANT)

class NativeLoopKernel(HummingKernel):
    def prepare(self):
        anchor = "#include <humming/kernel/humming.cuh>"
        assert self.code.count(anchor) == 1 and adapted_source() is not None
        self.code=self.code.replace(anchor,adapted_source())
        super().prepare()

@lru_cache(maxsize=256)
def kernel_handle(context, meta_json, compute_json, tuning_json, variant):
    global VARIANT
    VARIANT=variant
    config=json.loads(meta_json)
    config.pop("sublayer_name",None)
    config.update(json.loads(compute_json))
    config.update(json.loads(tuning_json))
    num_sms=config.pop("num_sms",0)
    kernel=NativeLoopKernel(**config)
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

def native_loop_forward(layer, **kwargs):
    if not eligible(kwargs) or adapted_source() is None:
        return HummingMethod.forward_layer(layer,**kwargs)
    meta=layer.humming_metas[kwargs["sublayer_name"]]
    if (meta.input_scale_group_size or meta.has_bias or meta.has_zero_point
            or str(meta.b_dtype)!="float4e2m1" or str(meta.bs_dtype)!="float8e4m3"
            or meta.weight_scale_group_size!=16):
        return HummingMethod.forward_layer(layer,**kwargs)
    handles=kernel_handle(HummingKernel.current_context(),meta.to_str(),
        json.dumps(kwargs["compute_config"],sort_keys=True),json.dumps(kwargs["tuning_config"],sort_keys=True),VARIANT)
    return ops.launch_kernel(
        configs=handles,inputs=kwargs["inputs"],outputs=kwargs["outputs"],
        weight=getattr(layer,meta.weight_name),weight_scale=getattr(layer,meta.weight_scale_name),
        weight_scale_2=getattr(layer,meta.weight_scale_2_name,None),
        sorted_ids=kwargs["sorted_ids"],expert_ids=kwargs["expert_ids"],
        num_tokens_padded=kwargs["num_tokens_padded"],locks=getattr(layer,"locks",None),
        top_k=kwargs["top_k"],valid_shape_m=kwargs["valid_shape_m"])
