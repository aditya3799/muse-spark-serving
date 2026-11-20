"""Target-only runtime ownership prototype. No controller, no serving hook.
Factories may use the pinned FA3 backend/DecodeCudaGraphRunner, but this module
never changes global speculative settings or constructs a second model/KV pool.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class TargetVerifyState:
    width: int
    backend: object
    graph: object


class TargetVerifyStates:
    def __init__(self, model_runner, draft_width, stream_key):
        self.runner = model_runner
        self.draft_width = int(draft_width)
        self.stream_key = stream_key
        self.thread = threading.get_ident()
        self.active = False
        self.startup_open = True
        self.invalidated_reason = None
        self.owners = self._owners()
        self.default_backend = model_runner.attn_backend
        self.default_graph = model_runner.decode_cuda_graph_runner
        self.states = {self.draft_width: TargetVerifyState(
            self.draft_width, self.default_backend, self.default_graph
        )}

    def _owners(self):
        r = self.runner
        return tuple(id(getattr(r, name)) for name in (
            'model', 'req_to_token_pool', 'token_to_kv_pool',
            'token_to_kv_pool_allocator', 'kv_index_translator',
            'prefill_cuda_graph_runner', 'graph_shared_output',
        ))

    def invalidate(self, reason):
        """Call before ANY weight mutation, recapture or topology replacement.
        Retain graph/buffer references until caller synchronizes and retires us.
        Owner identity alone cannot detect parameter.data or in-place updates.
        """
        if threading.get_ident() != self.thread or self.active:
            raise RuntimeError("cannot invalidate active/foreign-thread target states")
        self.invalidated_reason = str(reason)
        self.startup_open = False

    def bind_forward_stream(self, stream_key):
        # Scheduler creates its serial serving stream AFTER startup capture.
        self._guard()
        if self.startup_open:
            raise RuntimeError("bind serving stream only after startup capture")
        if self.stream_key is not None and self.stream_key != stream_key:
            raise RuntimeError("forward stream already bound")
        self.stream_key = stream_key

    def _guard(self):
        if self.invalidated_reason is not None:
            raise RuntimeError("target states invalidated: " + self.invalidated_reason)
        if threading.get_ident() != self.thread:
            raise RuntimeError('target state accessed from another host thread')
        if self._owners() != self.owners:
            raise RuntimeError('model/KV/input owner changed; recapture required')
        if self.active:
            raise RuntimeError('nested target state use is unsafe')
        if (self.runner.attn_backend is not self.default_backend
                or self.runner.decode_cuda_graph_runner is not self.default_graph):
            raise RuntimeError('target runner changed outside scoped state')

    def build(self, width, backend_factory, graph_factory):
        self._guard()
        if not self.startup_open:
            raise RuntimeError('capture after startup/KV sizing is forbidden')
        if not 1 <= width <= self.draft_width:
            raise ValueError('verify width outside fixed draft block')
        if width in self.states:
            return self.states[width]
        workspace = self.runner.init_new_workspace
        try:
            backend = backend_factory(self.runner)
        finally:
            self.runner.init_new_workspace = workspace
        if type(backend).__name__ != 'FlashAttentionBackend':
            raise RuntimeError('only an unwrapped FA3 backend is qualified')
        if backend.fa_impl_ver != 3 or self.runner.is_draft_worker:
            raise RuntimeError('target FA3 backend required')
        if backend.target_verify_metadata or backend._verify_mask is not None:
            raise RuntimeError('backend already owns width-dependent metadata')
        # Pinned target constructor reads width but has not allocated verify
        # tensors yet. This is a one-time binding BEFORE graph initialization.
        backend.speculative_num_draft_tokens = width
        self.runner.attn_backend = backend
        self.active = True
        try:
            graph = graph_factory(self.runner, backend, width)
            if graph.attn_backend is not backend or graph.captured_req_width != width:
                raise RuntimeError('captured graph/backend width mismatch')
            if graph.model_runner is not self.runner:
                raise RuntimeError('graph does not share the target model runner')
            state = TargetVerifyState(width, backend, graph)
        finally:
            self.runner.attn_backend = self.default_backend
            self.runner.decode_cuda_graph_runner = self.default_graph
            self.active = False
        self.states[width] = state
        return state

    def seal_startup(self):
        self._guard()
        self.startup_open = False

    @contextmanager
    def use(self, width, current_stream_key):
        self._guard()
        if current_stream_key != self.stream_key:
            raise RuntimeError('shared graph/input pool requires the forward stream')
        state = self.states.get(width)
        if state is None:
            raise ValueError('unbuilt verification width')
        self.active = True
        self.runner.attn_backend = state.backend
        self.runner.decode_cuda_graph_runner = state.graph
        try:
            # Scope must enclose preparation, forward, accept AND aux consumption.
            yield state
        finally:
            self.runner.attn_backend = self.default_backend
            self.runner.decode_cuda_graph_runner = self.default_graph
            self.active = False
