"""Start local projection after the target's packed auxiliary output becomes externally visible."""
import ast,difflib,hashlib,os,subprocess
from pathlib import Path
import overlap_build as base
R=base.ROOT;D=base.OUT
FILE=base.FILE
RUNNER="python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py"
LP="python/sglang/srt/layers/logits_processor.py"
orig={}
for file in (FILE,RUNNER,LP):
    assert ("diff --git a/"+file+" b/"+file).encode() not in base.parent
    orig[file]=subprocess.check_output(["git","-C",str(R/"sglang"),"show",base.PIN+":"+file]).decode()
sources=dict(orig)
sources[FILE]=base.new
def edit(f,a,b):
    assert sources[f].count(a)==1,(f,a)
    sources[f]=sources[f].replace(a,b)
edit(FILE,"    def _project_during_verify(self, hidden, rows, active):","    def _project_during_verify(self, hidden, rows, active, ready=None):")
edit(FILE,"        side.wait_stream(forward)","""        if (
            ready is not None and hidden.data_ptr() == ready[1].data_ptr()
            and hidden.stride() == ready[1].stride()
            and hidden.shape[0] <= ready[1].shape[0]
            and hidden.shape[1:] == ready[1].shape[1:]
        ):
            side.wait_event(ready[0])
        else:
            side.wait_stream(forward)""")
edit(FILE,"        with self._project_during_verify(\n            logits_output.hidden_states, bs * int(self.block_size),\n            active=deferred and can_run_cuda_graph,","""        ready = logits_output.dflash_context_ready
        logits_output.dflash_context_ready = None
        with self._project_during_verify(
            logits_output.hidden_states, bs * int(self.block_size),
            active=deferred and can_run_cuda_graph, ready=ready,""")
edit(RUNNER,"        # All setup hooks below read get_attn_backend()",'''        # Each captured shape owns the event recorded when its packed target
        # taps are ready. Keep concurrent stream variants on the ordinary fence.
        from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
            FullCudaGraphBackend,
        )

        if (
            isinstance(self.backend, FullCudaGraphBackend)
            and self.capture_forward_mode.is_target_verify()
            and self.model_runner.spec_algorithm.is_dflash_family()
            and not self.model_runner.is_draft_worker
            and not self.enable_pdmux and not self.ragged_verify_mode
            and self.pp_size == 1
        ):
            forward_batch.dflash_context_ready_event = make_external_event(self.device_module)

        # All setup hooks below read get_attn_backend()''')
edit(LP,"    def finalize_input_logprobs(self) -> None:","""    # A graph-local external event and the exact packed tensor it marks.
    dflash_context_ready: Optional[tuple] = None

    def finalize_input_logprobs(self) -> None:""")
edit(LP,"    draft_extend_select_index: Optional[torch.Tensor] = None\n\n    @classmethod","    draft_extend_select_index: Optional[torch.Tensor] = None\n    dflash_context_ready_event: Optional[torch.cuda.Event] = None\n\n    @classmethod")
edit(LP,"            capture_hidden_mode=forward_batch.capture_hidden_mode,","""            capture_hidden_mode=forward_batch.capture_hidden_mode,
            dflash_context_ready_event=getattr(forward_batch, "dflash_context_ready_event", None),""")
edit(LP,"            # Compute logits for both input and sampled tokens.\n            logits = self._get_logits","""            context_ready = None
            if (
                logits_metadata.dflash_context_ready_event is not None
                and self.defer_verify_gather
                and hidden_states_to_store is not None
                and hidden_states_to_store.ndim == 2
            ):
                event = logits_metadata.dflash_context_ready_event
                event.record()
                context_ready = (event, hidden_states_to_store)
            # Compute logits for both input and sampled tokens.
            logits = self._get_logits""")
edit(LP,"                next_token_logits=sampled_logits,\n                hidden_states=hidden_states_to_store,","""                next_token_logits=sampled_logits,
                hidden_states=hidden_states_to_store,
                dflash_context_ready=context_ready,""")
delta=[]
for f,s in sources.items():
    ast.parse(s)
    delta.append("diff --git a/"+f+" b/"+f+"\n")
    delta.extend(difflib.unified_diff(orig[f].splitlines(True),s.splitlines(True),"a/"+f,"b/"+f))
    (D/("event_"+Path(f).name)).write_text(s,encoding="utf-8")
out=D/"event_candidate.diff";full=base.parent+"".join(delta).encode()
assert max(len(x[1:]) for x in full.splitlines() if x.startswith(b"+"))<=400
out.write_bytes(full)
env={**os.environ,"GIT_INDEX_FILE":str(D/"event.index")}
for args in (["read-tree",base.PIN],["apply","--cached","--check",str(out)]):
    subprocess.run(["git","-C",str(R/"sglang"),*args],env=env,check=True)
print("event",len(full),hashlib.sha256(full).hexdigest())

