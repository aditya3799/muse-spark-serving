"""Stage indexed MoE read/write rows in a single collective load-thread pass."""
import json
from functools import lru_cache
from pathlib import Path

import torch
import humming
from humming import ops
from humming.kernel.humming import HummingKernel
from humming.layer import HummingMethod

OLD_BEGIN = "    const int4 *gmem_ptr_load = reinterpret_cast<const int4 *>(gmem_ptr);"
OLD_END = "    ctx.sync_load_threads();\n  };\n\n  CUDA_INLINE\n  void update_tensor_map_c"
FUSED = """    // Indexed output rows are already in slot order. Under the non-WS
    // scheduler, the epilogue synchronizes all threads after its last write.
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < CEIL_DIV(BlockShape::M, kNumLoadThreads); ++i) {
      const uint32_t idx_in_tile = i * kNumLoadThreads + threadIdx.x;
      if (idx_in_tile < BlockShape::M) {
        const uint32_t slot = gmem_ptr[idx_in_tile];
        ctx.smem.wr_row_index[idx_in_tile] = slot;
        ctx.smem.rd_row_index[idx_in_tile] = ctx.params.top_k == 1 ? slot : slot >> 3;
      }
    }

"""

@lru_cache(maxsize=1)
def adapted_source():
    directory = Path(humming.__file__).resolve().parent / "include/humming"
    scheduler = (directory/"scheduler.cuh").read_text()
    kernel = (directory/"kernel/humming.cuh").read_text()
    epilogue = (directory/"epilogue/pipeline.cuh").read_text()
    if (scheduler.count(OLD_BEGIN) != 1 or scheduler.count(OLD_END) != 1
            or kernel.count("#include <humming/scheduler.cuh>") != 1
            or "gmem_writer.write(slice_id, slice_count, i);\n      ctx.sync_math_threads();" not in epilogue):
        return None
    start = scheduler.index(OLD_BEGIN)
    end = scheduler.index(OLD_END,start)
    expected = scheduler[start:end]
    if ("legacy_load_1d<" not in expected or "ctx.smem.rd_row_index[index] = idx / ctx.params.top_k;" not in expected
            or expected.count("ctx.sync_load_threads()") != 1):
        return None
    return kernel.replace("#include <humming/scheduler.cuh>", scheduler[:start]+FUSED+scheduler[end:])

class StagedIndexKernel(HummingKernel):
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
    kernel=StagedIndexKernel(**config)
    return torch.tensor([kernel.kernel_id,num_sms],dtype=torch.int64,device="cpu")

def eligible(kwargs):
    x=kwargs["inputs"]
    c=kwargs["tuning_config"]
    compute=kwargs["compute_config"]
    return (kwargs["valid_shape_m"]>=4096 and kwargs["top_k"] in (1,8)
            and x.is_cuda and x.dtype==torch.bfloat16
            and isinstance(c,dict) and c.get("use_stream_k") is False
            and c.get("use_f16_accum") is False and not c.get("use_warp_spec")
            and not c.get("use_tma") and c.get("multi_cast_size_a",1)==1
            and c.get("multi_cast_size_b",1)==1
            and compute.get("gemm_type")=="indexed" and compute.get("use_f16_accum") is False)

def staged_index_forward(layer, **kwargs):
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
