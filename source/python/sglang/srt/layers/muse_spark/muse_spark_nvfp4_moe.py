"""Muse-specific gated alignment for FlashInfer's TRT-LLM FP4 MoE kernel."""

from __future__ import annotations

from typing import cast

import torch

from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptNvFp4FusedMoEMethod,
    _compute_gemm1_alphas,
)
from sglang.srt.layers.utils import copy_or_rebind_param


def _align_fp4_moe_weights_for_flashinfer_trtllm(layer) -> None:
    """Align Muse's ``[w1, w3]`` checkpoint layout in-place.

    The kernel permutation interleaves gated rows but does not exchange the
    checkpoint's logical gate/up halves. Muse therefore performs the explicit
    ``[w1, w3] -> [w3, w1]`` reorder before the standard shuffle. The aligned
    tensors replace the canonical parameters so the model keeps only one copy.
    """
    from sglang.srt.layers.quantization.utils import (
        prepare_static_weights_for_trtllm_fp4_moe,
        reorder_w1w3_to_w3w1,
    )
    from sglang.srt.layers.utils import copy_or_rebind_param

    w13_weight = cast(torch.Tensor, layer.gate_up_proj)
    w2_weight = cast(torch.Tensor, layer.down_proj)
    w13_weight_scale = cast(torch.Tensor, layer.gate_up_proj_weight_scale)
    w2_weight_scale = cast(torch.Tensor, layer.down_proj_weight_scale)

    if layer.moe_runner_config.is_gated:
        w13_weight, w13_weight_scale = reorder_w1w3_to_w3w1(
            w13_weight, w13_weight_scale
        )

    (
        gemm1_weights_fp4_shuffled,
        gemm1_scales_fp4_shuffled,
        gemm2_weights_fp4_shuffled,
        gemm2_scales_fp4_shuffled,
    ) = prepare_static_weights_for_trtllm_fp4_moe(
        w13_weight,
        w2_weight,
        w13_weight_scale,
        w2_weight_scale,
        w2_weight.size(-2),
        w13_weight.size(-2) // 2,
        w13_weight.size(0),
        is_gated=layer.moe_runner_config.is_gated,
    )

    copy_or_rebind_param(layer, "gate_up_proj", gemm1_weights_fp4_shuffled.contiguous())
    copy_or_rebind_param(layer, "down_proj", gemm2_weights_fp4_shuffled.contiguous())
    copy_or_rebind_param(
        layer, "gate_up_proj_weight_scale", gemm1_scales_fp4_shuffled.contiguous()
    )
    copy_or_rebind_param(
        layer, "down_proj_weight_scale", gemm2_scales_fp4_shuffled.contiguous()
    )

    # The up-projection (w3), not the gate (w1), supplies TRT-LLM's extra
    # GEMM1-output dequantization factor before GEMM2 requantization.
    w2_input_scale_quant = cast(torch.Tensor, layer.w2_input_scale_quant)
    w13_weight_scale_2 = cast(torch.Tensor, layer.gate_up_proj_weight_scale_2)
    if (
        layer.moe_runner_config.is_gated
        and w13_weight_scale_2.dim() == 2
        and w13_weight_scale_2.shape[1] >= 2
    ):
        w13_input_scale = (1.0 / layer.w13_input_scale_quant).to(torch.float32)
        g1_scale_c = (
            w13_input_scale * w13_weight_scale_2[:, 1] * w2_input_scale_quant
        ).to(torch.float32)
    else:
        g1_alphas = cast(torch.Tensor, layer.g1_alphas)
        g1_scale_c = (w2_input_scale_quant * g1_alphas).to(torch.float32)
    copy_or_rebind_param(layer, "g1_scale_c", g1_scale_c.contiguous())


class MuseSparkNvFp4FusedMoEMethod(ModelOptNvFp4FusedMoEMethod):
    """Prepare Muse FP4 experts without process-global mutation or duplication."""

    def __init__(self, quant_config):
        from sglang.srt.layers.moe import get_moe_runner_backend
        from sglang.srt.utils.common import get_device_capability, is_cuda

        backend = get_moe_runner_backend()
        if not backend.is_humming():
            super().__init__(quant_config)
            return
        if not is_cuda() or not (8, 0) <= get_device_capability() < (10, 0):
            raise ValueError("Muse NVFP4 Humming W4A16 requires SM80-SM90")
        self.quant_config = quant_config
        self._lora_marlin = False
        self.enable_flashinfer_trtllm_moe = False
        self.use_flashinfer_trtllm_weight_layout = False
        self._cache_permute_indices = {}
    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        """Allocate the serialized HF expert tensors without renaming them."""
        n, h, i = num_experts, hidden_size, intermediate_size_per_partition
        group = self.quant_config.group_size
        for name, shape, dtype in (
            ("gate_up_proj", (n, 2 * i, h // 2), torch.uint8),
            ("down_proj", (n, h, i // 2), torch.uint8),
            ("gate_up_proj_weight_scale", (n, 2 * i, h // group), torch.float8_e4m3fn),
            ("down_proj_weight_scale", (n, h, i // group), torch.float8_e4m3fn),
            ("gate_up_proj_weight_scale_2", (n, 2), torch.float32),
            ("down_proj_weight_scale_2", (n,), torch.float32),
            ("gate_up_proj_input_scale", (n, 2), torch.float32),
            ("down_proj_input_scale", (n,), torch.float32),
        ):
            layer.register_parameter(
                name,
                torch.nn.Parameter(
                    torch.empty(shape, dtype=dtype), requires_grad=False
                ),
            )

    # The base method's Marlin branch, and prepare_moe_nvfp4_layer_for_marlin
    # underneath it, address the experts by their pre-rename names.
    _MARLIN_PARAM_ALIASES = (
        ("w13_weight", "gate_up_proj"),
        ("w2_weight", "down_proj"),
        ("w13_weight_scale", "gate_up_proj_weight_scale"),
        ("w2_weight_scale", "down_proj_weight_scale"),
        ("w13_weight_scale_2", "gate_up_proj_weight_scale_2"),
        ("w2_weight_scale_2", "down_proj_weight_scale_2"),
    )

    def _process_weights_for_marlin(self, layer):
        for alias, native in self._MARLIN_PARAM_ALIASES:
            setattr(layer, alias, getattr(layer, native))
        result = super().process_weights_after_loading(layer)
        # Marlin repacks into freshly allocated parameters. Drop the native
        # names so the unpacked originals are freed instead of being kept
        # alive at full size alongside the repacked copies.
        for _, native in self._MARLIN_PARAM_ALIASES:
            if native in layer._parameters:
                del layer._parameters[native]
            elif hasattr(layer, native):
                delattr(layer, native)
        return result

    def _process_weights_for_humming(self, layer):
        from types import SimpleNamespace

        raw = SimpleNamespace()
        for alias, native in self._MARLIN_PARAM_ALIASES:
            setattr(raw, alias, getattr(layer, native))
        # Match stock Marlin's gate/up global-scale collapse exactly.
        if raw.w13_weight_scale_2.dim() > 1:
            raw.w13_weight_scale_2 = raw.w13_weight_scale_2[:, 0].contiguous()
        layer._humming_layer = _prepare_humming_experts(raw, torch)
        # Only the canonical Humming packed tensors remain resident.
        for _, native in self._MARLIN_PARAM_ALIASES:
            delattr(layer, native)

    def process_weights_after_loading(self, layer):
        """Prepare the TRT-LLM layout from native HF parameters in place."""
        backend = getattr(self, "_moe_runner_backend", None)
        if backend is not None and backend.is_humming():
            return self._process_weights_for_humming(layer)
        if backend is not None and backend.is_marlin():
            return self._process_weights_for_marlin(layer)
        input_scale = layer.gate_up_proj_input_scale.max().float()
        output_scale = layer.down_proj_input_scale.max().float()
        if self.quant_config.use_per_token_activation:
            input_scale = torch.ones_like(input_scale)
            output_scale = torch.ones_like(output_scale)
        gate_alpha, up_alpha = _compute_gemm1_alphas(
            layer.gate_up_proj_weight_scale_2, input_scale, True
        )
        copy_or_rebind_param(layer, "g1_alphas", gate_alpha)
        copy_or_rebind_param(layer, "g1_alphas_up", up_alpha)
        copy_or_rebind_param(
            layer, "g2_alphas", output_scale * layer.down_proj_weight_scale_2
        )
        copy_or_rebind_param(layer, "w13_input_scale_quant", input_scale.reciprocal())
        copy_or_rebind_param(layer, "w2_input_scale_quant", output_scale.reciprocal())
        for weight, scale in (
            (layer.gate_up_proj, layer.gate_up_proj_weight_scale),
            (layer.down_proj, layer.down_proj_weight_scale),
        ):
            if (
                scale.dtype != torch.float8_e4m3fn
                or scale.shape[-1] != weight.shape[-1] * 2 // 16
            ):
                raise ValueError(
                    "Muse NVFP4 requires one FP8 scale per 16 weight elements"
                )
        self._align_fp4_moe_weights_for_flashinfer_trtllm(layer)

    def _align_fp4_moe_weights_for_flashinfer_trtllm(self, layer) -> None:
        if getattr(layer, "use_megamoe", False):
            from sglang.srt.layers.muse_spark.muse_spark_megamoe import (
                build_nvfp4_megamoe_weights,
            )

            # Capture scale metadata while gate/up order is still canonical.
            build_nvfp4_megamoe_weights(layer)
        _align_fp4_moe_weights_for_flashinfer_trtllm(layer)
        if getattr(layer, "use_megamoe", False):
            from sglang.srt.layers.muse_spark.muse_spark_megamoe import (
                bind_megamoe_shared_weights,
            )

            bind_megamoe_shared_weights(layer, "nvfp4xnvfp4")


__all__ = [
    "MuseSparkNvFp4FusedMoEMethod",
    "_align_fp4_moe_weights_for_flashinfer_trtllm",
]


def _prepare_humming_experts(raw, torch):
    """Pack one canonical NVFP4 layout with FP8 block scales.

The caller validates aligned shapes, allowing repacking without redundant
zero-padding copies. Global scales are applied after each BF16 GEMM output
to preserve the stock Marlin rounding boundary. Tuning overrides are owned
by this layer; Humming's cached defaults remain unchanged."""
    from humming import dtypes
    from humming.config import WeightScale2Type, WeightScaleType
    from humming.layer import HummingMethod
    from humming.schema import HummingInputSchema, HummingWeightSchema
    layer = torch.nn.Module()
    for prefix in ('w13', 'w2'):
        w = getattr(raw, prefix + '_weight')
        sf = getattr(raw, prefix + '_weight_scale')
        gs = getattr(raw, prefix + '_weight_scale_2')
        e, n, k2 = w.shape
        if w.dtype != torch.uint8 or sf.dtype != torch.float8_e4m3fn:
            raise ValueError('Humming NVFP4 requires uint8 codes and FP8 scales')
        if n % 64 or k2 * 2 % 64 or sf.shape != (e, n, k2 * 2 // 16):
            raise ValueError('Humming NVFP4 requires aligned shapes and group size 16')
        if gs.ndim != 1 or gs.shape[0] != e:
            raise ValueError('Humming NVFP4 requires one global scale per expert')
        layer.register_parameter(prefix + '_weight', torch.nn.Parameter(w.contiguous().view(torch.int32), requires_grad=False))
        layer.register_parameter(prefix + '_weight_scale', torch.nn.Parameter(sf.contiguous(), requires_grad=False))
        layer.register_buffer(prefix + '_post_scale', gs.to(torch.bfloat16).contiguous())
        schema = HummingWeightSchema(b_dtype=dtypes.float4e2m1, bs_dtype=dtypes.float8e4m3, weight_scale_group_size=16, weight_scale_type=WeightScaleType.GROUP, weight_scale_2_type=WeightScale2Type.NONE)
        meta = HummingMethod.prepare_layer_meta(layer, shape_n=n, shape_k=k2 * 2, weight_schema=schema, input_schema=HummingInputSchema(a_dtype=dtypes.bfloat16), num_experts=e, torch_dtype=torch.bfloat16, sublayer_name=prefix)
        assert meta.a_dtype == dtypes.bfloat16 and meta.c_dtype == dtypes.bfloat16
        assert not meta.use_int_weight_scale and (not meta.has_input_scale)
        HummingMethod.transform_humming_layer(layer, sublayer_name=prefix, already_padded=True)
    layer.register_buffer('locks', torch.zeros(max(2048, torch.cuda.get_device_properties(0).multi_processor_count * 8), device='cuda', dtype=torch.int32))
    import copy
    layer.fp32_configs = {}
    for stage in ('w13', 'w2'):
        default = HummingMethod.get_default_tuning_configs(layer, use_f16_accum=False, gemm_type='indexed', sublayer_name=stage)
        snapshot = copy.deepcopy(default)
        owned = copy.deepcopy(default)
        for lo, hi, c in owned:
            # Stream-K reduces partial outputs in BF16 even with FP32 MMA.
            # Disable it to preserve the stock FP32 reduction contract.
            c['use_stream_k'] = False
            c['use_f16_accum'] = False
        assert default == snapshot, 'Default tuning cache must stay unchanged'
        layer.fp32_configs[stage] = owned
    from sglang.kernels.ops.moe.muse_spark_humming_split import prepare_large_prefill_partition
    prepare_large_prefill_partition(layer, torch)
    return layer


def _humming_expert_outputs_default(layer, x, local_ids, torch, retain=False, *, defer_downscale=False):
    """Compute local routes with FP32 accumulation and no Stream-K reduction.

First and final outputs alias one buffer as in the stock Marlin path.
The caller applies the original per-expert norm and weighted finalization."""
    from humming.layer import HummingMethod
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.muse_spark_humming_scale import scale_active_rows
    configs = layer.fp32_configs
    valid = local_ids.numel()
    selected = {s: next((c for lo, hi, c in cfg if lo < valid <= hi)) for s, cfg in configs.items()}
    assert all((c['use_stream_k'] is False and c['use_f16_accum'] is False for c in selected.values()))
    block = selected['w13']['block_shape'][0]
    sorted_ids, expert_ids, padded = moe_align_block_size(local_ids, block, layer.w13_post_scale.numel(), ignore_invalid_expert=True)
    m, k = x.shape
    topk = local_ids.shape[1]
    i = layer.humming_metas['w2'].shape_k
    cache13 = torch.empty((m * topk * max(2 * i, k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[:m * topk * 2 * i].view(m * topk, 2 * i)
    common = dict(sorted_ids=sorted_ids, expert_ids=expert_ids, num_tokens_padded=padded, valid_shape_m=valid, compute_config={'gemm_type': 'indexed', 'use_f16_accum': False})
    HummingMethod.forward_layer(layer, inputs=x, outputs=cache1, top_k=topk, sublayer_name='w13', tuning_config=selected['w13'], **common)
    # GEMM output has already rounded to BF16. Apply the BF16 global scale
    # in a separate BF16 operation, preserving Marlin's second rounding.
    scale_active_rows(cache1, local_ids, layer.w13_post_scale)
    cache2 = torch.empty((m * topk, i), device=x.device, dtype=x.dtype)
    silu_and_mul(cache1, cache2, expert_ids=local_ids.flatten(), expert_step=1)
    cache3 = cache13[:m * topk * k].view(m * topk, k)
    if selected['w2']['block_shape'][0] != block:
        second = moe_align_block_size(local_ids, selected['w2']['block_shape'][0], layer.w2_post_scale.numel(), ignore_invalid_expert=True)
        common.update(sorted_ids=second[0], expert_ids=second[1], num_tokens_padded=second[2])
    HummingMethod.forward_layer(layer, inputs=cache2, outputs=cache3, top_k=1, sublayer_name='w2', tuning_config=selected['w2'], **common)
    # Preserve the same BF16-output -> BF16-scale boundary for GEMM2.
    if not defer_downscale:
        scale_active_rows(cache3, local_ids, layer.w2_post_scale)
    return (cache3.view(m, topk, k), local_ids)


def _humming_expert_outputs(layer, x, local_ids, torch, retain=False, *,
                            defer_downscale=False, global_experts=None):
    from sglang.kernels.ops.moe.muse_spark_humming_split import split_expert_outputs
    return split_expert_outputs(
        _humming_expert_outputs_default,layer,x,local_ids,torch,global_experts,
        retain=retain,defer_downscale=defer_downscale,
    )
