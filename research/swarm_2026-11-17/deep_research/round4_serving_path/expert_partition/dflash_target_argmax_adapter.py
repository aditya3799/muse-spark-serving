"""Default-off research adapter for a target-only argmax CUDA graph tail.
Serial target worker / one current replay stream only. No active serving install.
"""
import torch
from sglang.srt.runtime_context import get_disagg, get_exec, get_parallel
from sglang.srt.speculative.dflash_target_argmax_capture import TargetArgmaxCapture

RESEARCH_ENABLE_TARGET_ARGMAX_CAPTURE = False


def _topology_key():
    p = get_parallel()
    g = p.tp_group
    return (id(g), tuple(g.ranks), g.rank_in_group, g.world_size)


def _supported_worker(worker):
    p = get_parallel()
    mr = worker.target_worker.model_runner
    return (
        type(worker).__name__ == 'DFlashWorkerV2'
        and torch.device(worker.device).type == 'cuda'
        and int(worker.block_size) == 8
        and p.tp_size == 8 and p.tp_group.world_size == 8
        and not p.attn_dp_enabled
        and p.attn_cp_size == 1 and p.attn_dcp_size == 1 and p.pp_size == 1
        and not get_disagg().enable_pdmux
        and not get_exec().overlap.enable_two_batch_overlap
        and not mr.is_draft_worker
        and not mr.model_config.is_encoder_decoder
        and mr.lora_manager is None
        and not worker._is_domino
        and worker._target_logits_processor is not None
        and worker._target_logits_processor.defer_verify_gather
    )


def _lp_key(lp):
    # Compare only scalar Python values, never tensors or overloaded equality.
    cap = lp.final_logit_softcapping
    vocab = lp.vocab_size
    deferred = lp.defer_verify_gather
    if ((cap is not None and type(cap) not in (int, float))
            or type(vocab) is not int or type(deferred) is not bool):
        return None
    if cap is not None and not (-float('inf') < cap < float('inf')):
        return None
    return (id(lp), cap, vocab, deferred)


def _plain_key(key):
    return (
        key is not None and key.size > 0
        and key.stream_idx is None and key.variant_label is None
        and key.attention_variant is None
    )


class ResearchTargetArgmaxAdapter:
    def __init__(self, worker):
        self.worker = worker
        self.model_runner = worker.target_worker.model_runner
        self.state = None
        self.runner = None
        self.topology = None
        self.lp_key = None
        self.forward_stream_key = None
        self.confirmed_graphs = {}
        self.disabled_reason = None

    def _supported_runner(self, runner):
        # Every field is source-backed; unknown future schema rejects explicitly.
        try:
            return (
                _supported_worker(self.worker)
                and runner.model_runner is self.model_runner
                and type(runner).__name__ == 'DecodeCudaGraphRunner'
                and type(runner.backend).__name__ == 'FullCudaGraphBackend'
                and runner.capture_forward_mode.is_target_verify()
                and runner.captured_req_width == int(self.worker.block_size)
                and not runner.require_gathered_buffer
                and not runner.require_mlp_sync
                and not runner.enable_two_batch_overlap
                and not runner.enable_prefill_cp
                and not runner.enable_pdmux
                and not runner.record_nolora_graph
                and runner.attention_graph_variants is None
                and not runner.ragged_verify_mode
                and not runner.is_dllm
                and self.model_runner.lora_manager is None
            )
        except (AttributeError, TypeError):
            return False

    def _invalidate(self):
        if self.state is not None:
            self.state.invalidate()
        self.confirmed_graphs.clear()
        self.runner = None
        self.topology = None
        self.lp_key = None
        self.forward_stream_key = None

    def on_capture_start(self, runner):
        self._invalidate()
        if not self._supported_runner(runner):
            self.disabled_reason = 'unsupported_runner'
            return
        lp_key = _lp_key(self.worker._target_logits_processor)
        if lp_key is None:
            self.disabled_reason = 'unsupported_logits_processor_scalars'
            return
        rows = runner.max_bs * runner.captured_req_width
        if self.state is None:
            self.state = TargetArgmaxCapture(
                self.worker._target_logits_processor, max_rows=rows,
                width=runner.captured_req_width, device=self.worker.device)
        elif rows > self.state.output.numel():
            # Never free or replace a buffer that an old graph could still write.
            self.disabled_reason = 'recapture_capacity_growth_requires_new_adapter'
            return
        if self.state.lp is not self.worker._target_logits_processor:
            self.disabled_reason = 'logits_processor_changed'
            return
        self.runner = runner
        self.topology = _topology_key()
        self.lp_key = lp_key
        self.forward_stream_key = self.model_runner.forward_stream.cuda_stream
        self.disabled_reason = None

    def on_capture_abort(self, runner):
        self._invalidate()
        self.disabled_reason = 'capture_aborted'

    def __call__(self, runner, out, forward_batch, num_tokens):
        if (runner is not self.runner or not self._supported_runner(runner)
                or _topology_key() != self.topology
                or _lp_key(self.worker._target_logits_processor) != self.lp_key):
            return
        self.state.capture_hook(runner, out, forward_batch, num_tokens)

    def on_capture_success(self, runner, forward_batch, num_tokens, shape_key):
        if (runner is not self.runner or not self._supported_runner(runner)
                or _topology_key() != self.topology
                or _lp_key(self.worker._target_logits_processor) != self.lp_key
                or not _plain_key(shape_key)
                or shape_key.size * runner.captured_req_width != num_tokens
                or not forward_batch.forward_mode.is_target_verify()
                or shape_key not in runner.backend._graphs):
            return
        if self.state.confirm_capture(runner=runner, captured_rows=num_tokens,
                                      generation=self.state.generation):
            self.confirmed_graphs[shape_key] = runner.backend._graphs[shape_key]

    def consume(self, *, raw_rows, graph_used, can_skip_verify_logits):
        runner = self.model_runner.decode_cuda_graph_runner
        if (not graph_used or not can_skip_verify_logits
                or runner is not self.runner or not self._supported_runner(runner)
                or _topology_key() != self.topology
                or _lp_key(self.worker._target_logits_processor) != self.lp_key):
            return None
        # Capture uses a private stream; never compare with that stream.
        # Scored overlap execution calls target and this consumer inside the
        # target model runner's stable forward_stream context. Other streams
        # fall back, including unsupported non-overlap/default-stream use.
        if (self.model_runner.forward_stream.cuda_stream != self.forward_stream_key
                or torch.cuda.current_stream(self.worker.device).cuda_stream
                != self.forward_stream_key):
            return None
        key = getattr(runner, '_replay_graph_key', None)
        if (not _plain_key(key) or key.size != runner.bs
                or runner.raw_num_token != raw_rows
                or key not in self.confirmed_graphs
                or runner.backend._graphs.get(key) is not self.confirmed_graphs[key]):
            return None
        return self.state.consume(
            runner=runner, raw_rows=raw_rows,
            captured_rows=runner.bs * runner.captured_req_width,
            width=int(self.worker.block_size), graph_used=graph_used,
            can_skip_verify_logits=can_skip_verify_logits,
            generation=self.state.generation)


def maybe_install_target_argmax_capture(worker, *, enabled=None):
    if enabled is None:
        enabled = RESEARCH_ENABLE_TARGET_ARGMAX_CAPTURE
    if not enabled:
        return None
    try:
        supported = _supported_worker(worker)
    except (AttributeError, TypeError):
        supported = False
    if not supported:
        return None
    # Registration must precede the target runner's first CUDA graph capture.
    mr = worker.target_worker.model_runner
    if getattr(mr, 'decode_cuda_graph_runner', None) is not None:
        return None
    adapter = ResearchTargetArgmaxAdapter(worker)
    mr.capture_tail_hooks.append(adapter)
    return adapter

