"""Default-off target-only width graph lifecycle prototype. No runtime imports.
Pattern follows EAGLE adaptive fresh attention backend + per-width graph runner.
No draft state or model weights are modified. CPU tests do not qualify a width.
"""
from contextlib import contextmanager
from dataclasses import dataclass,field
REQUIRED_GATES=frozenset(('attention_prefix_bitwise','full_model_greedy','capture_replay_dynamic','memory_latency_limits'))

@dataclass(frozen=True)
class TargetState:
    width:int
    backend:object
    graph:object
    qualification:frozenset=field(default_factory=frozenset)
    allocated_delta_bytes:object=None


def build_target_state(model_runner,graph_runner_cls,width,capture_scope):
    """Startup-only factory. capture_scope temporarily overrides target spec width.
    The real adapter must use get_context().override like EAGLE capture, restore
    in finally, and run all TP ranks together before serving begins.
    """
    if width not in (1,2,4,6):raise ValueError('Unsupported width')
    old_backend=model_runner.attn_backend
    old_graph=model_runner.decode_cuda_graph_runner
    old_workspace=model_runner.init_new_workspace
    try:
        with capture_scope(width):
            backend=model_runner._get_attention_backend(init_new_workspace=True)
            model_runner.init_new_workspace=old_workspace
            graph=graph_runner_cls(model_runner,attn_backend=backend,speculative_num_draft_tokens=width)
            if graph.captured_req_width!=width:raise RuntimeError('Graph captured wrong width')
            return TargetState(width,backend,graph)
    finally:
        model_runner.init_new_workspace=old_workspace
        model_runner.attn_backend=old_backend
        model_runner.decode_cuda_graph_runner=old_graph

class Lease:
    def __init__(self,registry,state):
        self.registry=registry;self.state=state;self.alive=True;self.aux_pending=False
    def mark_target_complete(self):
        if not self.alive or self.aux_pending:raise RuntimeError('Invalid target output lifetime')
        self.aux_pending=True
    def consume_aux(self):
        if not self.alive or not self.aux_pending:raise RuntimeError('No live target auxiliary output')
        self.aux_pending=False

class TargetGraphRegistry:
    def __init__(self,model_runner):
        self.runner=model_runner
        self.states={8:TargetState(8,model_runner.attn_backend,model_runner.decode_cuda_graph_runner,REQUIRED_GATES)}
        self.active=None
    def register(self,state):
        if self.active is not None:raise RuntimeError('Cannot capture/register during replay')
        if state.width==8 or state.width in self.states:raise ValueError('Cannot replace captured state')
        if state.graph.captured_req_width!=state.width:raise ValueError('Mismatched state')
        if state.backend is self.states[8].backend:raise ValueError('Attention metadata workspace must be separate')
        self.states[state.width]=state
    def choose(self,width):
        state=self.states.get(width)
        return state if state is not None and REQUIRED_GATES<=state.qualification else self.states[8]
    @contextmanager
    def activate(self,width):
        # This is a single scheduler-owner contract, not a cross-thread lock.
        # Integration must hold exclusive runner ownership through aux consume.
        if self.active is not None:raise RuntimeError('Concurrent or nested graph lease')
        state=self.choose(width);lease=Lease(self,state)
        previous=(self.runner.attn_backend,self.runner.decode_cuda_graph_runner)
        self.active=lease
        self.runner.attn_backend=state.backend;self.runner.decode_cuda_graph_runner=state.graph
        try:
            yield lease
            if lease.aux_pending:raise RuntimeError('Cannot release before target hidden materialization')
        finally:
            self.runner.attn_backend,self.runner.decode_cuda_graph_runner=previous
            lease.alive=False;self.active=None

def known_input_buffer_bytes(max_bs,width,hidden_size,aux_width,element_size=2,seq_fill=1,cache_id_bytes=8):
    """Unshared requested DecodeInputBuffers bytes, excluding unknown workspaces.
    Logits reuse existing full8 shared-output capacity; different-size input/aux
    tensors do NOT share by the pinned (name,numel,dtype,device) pool key.
    """
    rows=max_bs*width
    return rows*(8+cache_id_bytes+8+24+element_size*(hidden_size+aux_width))+max_bs*16+(max_bs*seq_fill+rows)*width+12
