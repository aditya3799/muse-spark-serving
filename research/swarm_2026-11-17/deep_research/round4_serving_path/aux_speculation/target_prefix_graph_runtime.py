"""Default-off target-only graph factory/scoping scaffold, not production enabled.
Requires explicit pre-mutation invalidation integration before production use.
"""
from contextlib import contextmanager
from target_verify_runtime_state import TargetVerifyStates

RESEARCH_TARGET_VERIFY_WIDTHS = ()


def supported_startup(worker, runtime):
    """Reject unsupported execution before constructing a second backend."""
    try:
        mr=worker.target_worker.model_runner
        p=runtime.parallel; e=runtime.execution; d=runtime.disagg
        graph=mr.decode_cuda_graph_runner
        return (
            int(worker.block_size)==8 and mr.spec_algorithm.is_dflash()
            and not mr.is_draft_worker and not mr.model_config.is_encoder_decoder
            and type(mr.attn_backend).__name__=='FlashAttentionBackend'
            and mr.attn_backend.fa_impl_ver==3 and mr.attn_backend.topk<=1
            and type(graph).__name__=='DecodeCudaGraphRunner'
            and type(graph.backend).__name__=='FullCudaGraphBackend'
            and graph.captured_req_width==8
            and not graph.require_gathered_buffer and not graph.require_mlp_sync
            and not graph.attention_graph_variants and not graph.ragged_verify_mode
            and not graph.enable_two_batch_overlap and not graph.enable_pdmux
            and not graph.enable_prefill_cp and not graph.record_nolora_graph
            and not graph.is_dllm and graph.capture_forward_mode.is_target_verify()
            and not p.attn_dp_enabled and p.attn_cp_size==1 and p.attn_dcp_size==1
            and p.pp_size==1 and not d.enable_pdmux
            and not e.overlap.enable_two_batch_overlap
            and mr.lora_manager is None and not worker._need_mamba_verify_commit
            and worker.lilicorr is None and not worker._is_domino
            and not mr.capture_tail_hooks
        )
    except (AttributeError,TypeError):
        return False


def initialize_target_prefix_states(worker, *, widths=None, runtime=None,
                                    backend_factory=None, graph_factory=None):
    """Call at DFlash.init_cuda_graphs entry, BEFORE entering draft TP scope.
    Target startup graph exists; KV post-capture resize/prewarm has not run.
    This allocates no weights or KV pools. Default-off returns immediately.
    """
    if widths is None: widths=RESEARCH_TARGET_VERIFY_WIDTHS
    if not widths: return None
    if getattr(worker,'_prefix_target_states',None) is not None:
        raise RuntimeError('prefix states already initialized')
    if runtime is None:
        from types import SimpleNamespace
        from sglang.srt.runtime_context import get_parallel,get_exec,get_disagg
        runtime=SimpleNamespace(parallel=get_parallel(),execution=get_exec(),disagg=get_disagg())
    if not supported_startup(worker,runtime):
        raise RuntimeError('unsupported target-only prefix graph topology')
    if any(type(w) is not int or w not in (1,2,4,6,8) for w in widths):
        raise ValueError('unsupported fixed target verification width')
    mr=worker.target_worker.model_runner
    if backend_factory is None:
        backend_factory=lambda r:r._get_attention_backend(init_new_workspace=True)
    if graph_factory is None:
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
        def graph_factory(r,b,w):
            # Explicit width override; global/draft config remains8.
            return DecodeCudaGraphRunner(r,attn_backend=b,speculative_num_draft_tokens=w)
    manager=TargetVerifyStates(mr,8,None)
    try:
        for width in widths: manager.build(width,backend_factory,graph_factory)
        manager.seal_startup()
    except BaseException:
        manager.invalidate('startup capture failure')
        # Keep partial captures alive. Startup must abort, not continue serving.
        worker._prefix_target_states_failed=manager
        raise
    worker._prefix_target_states=manager
    return manager


@contextmanager
def target_prefix_scope(worker, width, *, stream_key=None):
    manager=getattr(worker,'_prefix_target_states',None)
    if manager is None:
        yield None
        return
    if stream_key is None:
        import torch
        stream_key=torch.cuda.current_stream(worker.device).cuda_stream
    manager.bind_forward_stream(stream_key)
    with manager.use(width,stream_key) as state:
        yield state


def invalidate_target_prefix_states(worker, reason):
    manager=getattr(worker,'_prefix_target_states',None)
    if manager is not None: manager.invalidate(reason)