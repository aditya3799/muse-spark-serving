"""Opt-in model hook. Every unsupported state returns None before a KV write."""
import os
import torch
from .runtime import Fa3Prologue

def maybe_fuse(attention,qkv,positions,forward_batch):
    if os.environ.get('SGLANG_EXPERIMENTAL_MUSE_FA3_PROLOGUE','0')!='1':return None
    from sglang.kernels.fused_op import get_fused_op_backend
    from sglang.kernels.spec import KernelBackend
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
    from sglang.srt.layers.cp.utils import is_cp_active
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.runtime_context import get_memory,get_parallel
    backend=get_attn_backend();norm=attention._fused_qk_norm
    if norm is None or norm.is_torch_compile:return None
    if get_fused_op_backend() not in (None,KernelBackend.TRITON):return None
    if norm.auto_selected_backend()!=KernelBackend.TRITON:return None
    if type(backend) is not FlashAttentionBackend or backend.fa_impl_ver!=3:return None
    if backend.is_draft_runner or backend.topk>1:return None
    if get_memory().enable_unified_memory or get_parallel().attn_dp_enabled:return None
    if is_cp_active(forward_batch) or forward_batch.dcp_kv_mask is not None:return None
    mode=forward_batch.forward_mode
    if not (mode.is_decode() or mode.is_target_verify()):return None
    if backend.kv_cache_dtype!=torch.float8_e4m3fn:return None
    if attention.attn.k_scale is not None or attention.attn.v_scale is not None:return None
    buffers=backend.get_fused_kv_write_buffers(attention.attn,forward_batch,allow_fp8_prologue=True)
    if buffers is None:return None
    k,v,loc=buffers
    if k.ndim!=3 or v.shape!=k.shape or k.shape[1:]!=(2,64):return None
    if not k.is_contiguous() or not v.is_contiguous():return None
    rope=attention.rotary_emb.cos_sin_cache if attention.rotary_emb is not None else None
    pos=positions if rope is not None else None
    if not norm.backend_eligible(KernelBackend.TRITON,qkv,pos,rope,kv_store=None):return None
    adapter=getattr(attention,'_experimental_fa3_prologue',None)
    if adapter is None:
        if torch.cuda.is_current_stream_capturing():return None
        adapter=Fa3Prologue()
        attention._experimental_fa3_prologue=adapter
    return adapter.try_run(qkv,pos,rope,k.view(k.shape[0],128),v.view(v.shape[0],128),loc,
        reference=lambda x,p,r:norm._forward_triton(x,p,r,kv_store=None),
        head_dim=attention.head_dim,num_q_heads=attention.num_heads,num_kv_heads=attention.num_kv_heads,
        eps=attention.norm_eps,k_scale=attention.attn.k_scale,v_scale=attention.attn.v_scale,
        reference_is_triton=True,compatible_pool=True)
