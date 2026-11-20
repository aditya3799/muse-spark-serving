"""Create an isolated cumulative source-only experimental patch against the exact scored parent."""
import ast
import hashlib
import os
from pathlib import Path
import subprocess
from difflib import unified_diff

here=Path(__file__).resolve().parent
root=here.parent
repo=root/"sglang"
base="297e54fd105e974cdf401e5b4d20253a22c16f90"
parent=(root/"release_2026-11-18/candidate.diff").read_bytes()
assert hashlib.sha256(parent).hexdigest()=="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
out=here/"candidate.diff"
pieces=[]
def apply(name,changes):
    old=subprocess.check_output(["git","-C",str(repo),"show",base+":"+name]).decode()
    new=old
    for a,b in changes:
        assert new.count(a)==1,a
        new=new.replace(a,b)
    assert ("diff --git a/"+name+" b/"+name).encode() not in parent
    stage(name,old,new)
def stage(name,old,new):
    ast.parse(new)
    s="diff --git a/"+name+" b/"+name+"\n"
    if not old: s+="new file mode 100644\n"
    s+="".join(unified_diff(old.splitlines(True),new.splitlines(True),("a/"+name if old else "/dev/null"),"b/"+name))
    pieces.append(s)
    dst=here/"applied"/name
    dst.parent.mkdir(exist_ok=True,parents=True)
    dst.write_text(new,encoding="utf8")
module="python/sglang/srt/speculative/dflash_window_namespaces.py"
stage(module,"",(here/"namespaces.py").read_text())
scheduler="python/sglang/srt/managers/scheduler.py"
init="        self.tree_cache = result.tree_cache\n"
assign="    def _add_request_to_queue(self, req: Req, is_retracted: bool = False):\n"
apply(scheduler,[(init,init+'''        builder = getattr(self.draft_worker, "build_window_namespaces", None)
        self._draft_window_namespaces = None
        if (builder is not None and get_disagg().disaggregation_mode == "null"
            and not self.enable_hicache_storage and get_parallel().pp_size == 1):
            self._draft_window_namespaces = builder(self.page_size)
'''),(assign,assign+'''        if self._draft_window_namespaces is not None:
            self._draft_window_namespaces.assign(req)
''')])
worker="python/sglang/srt/speculative/dflash_worker_v2.py"
method="    def init_cuda_graphs(self):\n"
point="            # Materialize prompt tokens into the draft KV cache immediately."
apply(worker,[(method,'''    def build_window_namespaces(self, page_size):
        from sglang.srt.speculative.dflash_window_namespaces import DFlashWindowNamespaces

        if (self._is_domino or self.lilicorr is not None or get_parallel().attn_dp_enabled
            or not is_cuda()):
            return None
        config = self.draft_model.config
        layers = getattr(config, "layer_types", None)
        window = getattr(config, "sliding_window", None)
        if (not layers or not all(kind == "sliding_attention" for kind in layers)
            or len(layers) != len(self.draft_model.layers) or not window or window <= 0):
            return None
        self._window_namespaces = DFlashWindowNamespaces(window, page_size)
        return self._window_namespaces

'''+method),(point,'''            namespaces = getattr(self, "_window_namespaces", None)
            if (namespaces is not None and batch.forward_mode == ForwardMode.EXTEND
                and not batch.is_extend_in_batch
                and namespaces.can_omit(batch.reqs, batch.prefix_lens, batch.extend_lens)):
                # Every later request that can reuse these slots has at least its
                # namespace's floor tokens before its first draft read. No layer
                # can attend to this entire batch. Preserve the target prefill
                # and its logits, and let the scheduler publish as usual.
                logits_output.hidden_states = None
                batch_output.next_draft_input = self._make_next_draft_input_prefill(
                    bonus_tokens=next_token_ids,
                    seq_lens=new_seq_lens,
                )
                return batch_output

'''+point)])
payload=parent+"".join(pieces).encode()
assert len(payload)<=400000 and max(len(s[1:]) for s in payload.decode().splitlines() if s.startswith("+"))<=400
out.write_bytes(payload)
index=here/"window.index"
env={**os.environ,"GIT_INDEX_FILE":str(index)}
subprocess.run(["git","-C",str(repo),"read-tree",base],env=env,check=True)
subprocess.run(["git","-C",str(repo),"apply","--cached","--check",str(out)],env=env,check=True)
print(len(payload),hashlib.sha256(payload).hexdigest())

