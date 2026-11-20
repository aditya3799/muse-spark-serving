"""Model-call-site prototype. Enable only after the local/Hopper equality gates pass."""
import torch

def try_projection(attn,x,g,forward_batch):
    from sglang.kernels.fused_op import get_fused_op_backend
    from sglang.kernels.spec import KernelBackend
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod,get_bf16_gemm_backend
    from sglang.srt.layers.linear import use_symmetric_memory,should_skip_mlp_all_reduce,get_group_rank_size
    from sglang.srt.runtime_context import get_forward,get_parallel
    from kernel import eligible,project
    mode=forward_batch.forward_mode
    op=attn._fused_o_norm_gate;linear=attn.o_proj
    if not (mode.is_decode() or mode.is_target_verify()) or torch.compiler.is_compiling() or not x.is_cuda:
        return None
    if not (op._use_o_norm and op._use_o_gate and op._o_norm is not None and not op._o_norm.has_weight):
        return None
    forced=get_fused_op_backend()
    if forced not in (None,KernelBackend.JIT) or (forced is None and op.auto_selected_backend()!=KernelBackend.JIT):
        return None
    if not op.backend_eligible(KernelBackend.JIT,x,g) or linear.bias is not None:
        return None
    if type(linear.quant_method) is not UnquantizedLinearMethod or get_bf16_gemm_backend().value not in ("auto","torch"):
        return None
    if (not linear.reduce_results or not linear.use_dp_attention_reduce or not linear.input_is_parallel
            or linear.parallel_group!="attn_tp" or get_forward().sp_active or should_skip_mlp_all_reduce()):
        return None
    group=get_parallel().attn_tp_group
    _,tp=get_group_rank_size(linear.tp_group)
    if tp<=1 or torch.cuda.get_device_capability(x.device)[0]!=9:
        return None
    if not eligible(x,g,linear.weight,attn.head_dim,torch.cuda.get_device_properties(x.device).multi_processor_count):
        return None
    with use_symmetric_memory(group):
        partial=project(x,g,linear.weight,op._eps)
    # Exact collective entry and BF16 boundary used by RowParallelLinear.forward.
    return group.all_reduce(partial)

