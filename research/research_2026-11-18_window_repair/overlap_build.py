"""Prepare the exact source delta for concurrent local projection and verifier selection."""
import ast
import difflib
import hashlib
import os
import subprocess
import textwrap
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/"decode_overlap"
OUT.mkdir(exist_ok=True)
ROOT=HERE.parent
PIN="297e54fd105e974cdf401e5b4d20253a22c16f90"
BASE="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
FILE="python/sglang/srt/speculative/dflash_worker_v2.py"
old=subprocess.check_output(["git","-C",str(ROOT/"sglang"),"show",PIN+":"+FILE]).decode()
parent=(ROOT/"release_2026-11-18/candidate.diff").read_bytes()
assert hashlib.sha256(parent).hexdigest()==BASE and ("diff --git a/"+FILE+" b/"+FILE).encode() not in parent
new=old
def replace(a,b):
    global new
    assert new.count(a)==1, a
    new=new.replace(a,b)
replace("import os\n","import os\nfrom contextlib import contextmanager\n")
replace("        self._target_logits_processor = self._defer_target_verify_gather()","""        self._target_logits_processor = self._defer_target_verify_gather()
        self._context_projection_stream = self._new_context_projection_stream()""")
original="    def _defer_target_verify_gather(self)"
replace(original, '''    def _new_context_projection_stream(self):
        from sglang.srt.layers.layernorm import RMSNorm
        from sglang.srt.models.dflash import DFlashDraftModel

        model = self.draft_model
        if (
            self._target_logits_processor is None
            or not is_cuda()
            or type(model).project_target_hidden is not DFlashDraftModel.project_target_hidden
            or model.quantizable
            or type(model.fc) is not torch.nn.Linear
            or type(model.hidden_norm) is not RMSNorm
            or model.hidden_norm._quant_linear is not None
            or model.fc.weight.dtype != torch.bfloat16
            or model.fc.weight.device.type != "cuda"
        ):
            return None
        return torch.cuda.Stream(device=model.fc.weight.device)

    @contextmanager
    def _project_during_verify(self, hidden, rows, active):
        side = self._context_projection_stream
        if (
            not active or side is None or hidden is None
            or hidden.ndim != 2 or hidden.shape[0] != rows or rows == 0
            or hidden.shape[1] != self.draft_model.fc.in_features
            or hidden.device != self.draft_model.fc.weight.device
            or hidden.dtype != torch.bfloat16
            or torch.cuda.is_current_stream_capturing()
        ):
            yield None
            return
        forward = torch.cuda.current_stream(hidden.device)
        side.wait_stream(forward)
        try:
            with torch.cuda.stream(side), torch.inference_mode(), draft_tp_context(self.draft_parallel_layout):
                hidden.record_stream(side)
                projected = self.draft_model.project_target_hidden(hidden)
            # The following prefix-KV operation consumes the side allocation on
            # forward. The graph runner reuses the target aux buffer next round.
            projected.record_stream(forward)
            yield projected
        finally:
            forward.wait_stream(side)

''' + original)
replace("        extend_lens: Optional[torch.Tensor] = None,\n    ) -> None:\n        \"\"\"Materialize target context",
        "        extend_lens: Optional[torch.Tensor] = None,\n        projected_context: Optional[torch.Tensor] = None,\n    ) -> None:\n        \"\"\"Materialize target context")
replace("            ctx_hidden = self.draft_model.project_target_hidden(target_hidden)","""            ctx_hidden = (
                self.draft_model.project_target_hidden(target_hidden)
                if projected_context is None else projected_context
            )""")
start=new.index("        if target_lp is not None and target_lp.is_deferred_shard(")
stop=new.index("        # --- 3) Materialize committed",start)
body=new[start:stop]
source_if="""        if target_lp is not None and target_lp.is_deferred_shard(
            logits_output.next_token_logits
        ):"""
assert body.startswith(source_if)
body=body.replace(source_if,"        if deferred:",1)
head="""        deferred = target_lp is not None and target_lp.is_deferred_shard(
            logits_output.next_token_logits
        )
        with self._project_during_verify(
            logits_output.hidden_states, bs * int(self.block_size),
            active=deferred and can_run_cuda_graph,
        ) as projected_context:
"""
new=new[:start]+head+textwrap.indent(body,"    ")+new[stop:]
replace("            commit_lens=commit_lens,\n        )\n\n        # Avoid copying large hidden-state buffers","            commit_lens=commit_lens,\n            projected_context=projected_context,\n        )\n\n        # Avoid copying large hidden-state buffers")
ast.parse(new)
delta="diff --git a/"+FILE+" b/"+FILE+"\n"+"".join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),"a/"+FILE,"b/"+FILE))
(OUT/"delta.diff").write_bytes(delta.encode())
(OUT/"worker.py").write_text(new,encoding="utf-8")
full=parent+delta.encode()
assert max(len(x[1:]) for x in full.splitlines() if x.startswith(b"+")) <=400
path=OUT/"candidate.diff";path.write_bytes(full)
env={**os.environ,"GIT_INDEX_FILE":str(OUT/"apply.index")}
for args in (["read-tree",PIN],["apply","--cached","--check",str(path)]):
    subprocess.run(["git","-C",str(ROOT/"sglang"),*args],env=env,check=True)
print(len(full),hashlib.sha256(full).hexdigest())

