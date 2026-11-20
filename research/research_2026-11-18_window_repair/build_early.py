"""Cumulative early target-tap omission on the previously checked region candidate."""
from pathlib import Path
import subprocess,os,difflib,hashlib,ast
base=Path(__file__).resolve().parent
root=base.parent;repo=root/"sglang";pin="297e54fd105e974cdf401e5b4d20253a22c16f90"
env={**os.environ,"GIT_INDEX_FILE":str(base/"early.index")}
def git(*args,**kw):return subprocess.check_output(["git","-C",str(repo),*args],env=env,**kw)
git("read-tree",pin)
git("apply","--cached",str(base/"region_candidate.diff"))
def modify(name,replacements):
    old=git("show",":"+name).decode();new=old
    for a,b in replacements:
        assert new.count(a)==1,a
        new=new.replace(a,b)
    ast.parse(new)
    delta="".join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),"a/"+name,"b/"+name))
    git("apply","--cached","--whitespace=nowarn","-",input=delta.encode())
    dest=base/"applied_early"/name
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(new,encoding="utf8")
tp="python/sglang/srt/managers/tp_worker.py"
modify(tp,[("        return_kv_loc_plan: bool = False,\n    ) -> GenerationBatchResult:","        return_kv_loc_plan: bool = False,\n        dflash_skip_prefill_aux: bool = False,\n    ) -> GenerationBatchResult:"),
    ("            if folded is not None:\n                apply_folded_extend",'''            if dflash_skip_prefill_aux:
                forward_batch.dflash_skip_prefill_aux = True
            if folded is not None:
                apply_folded_extend''')])
muse="python/sglang/srt/models/muse_spark.py"
modify(muse,[("class MuseSparkTextForCausalLM(nn.Module):\n","class MuseSparkTextForCausalLM(nn.Module):\n    supports_dflash_window_aux_omission = True\n"),("        aux_hidden_states = []\n        for layer_idx","        tap_layers = () if getattr(forward_batch, \"dflash_skip_prefill_aux\", False) else self.layers_to_capture\n        aux_hidden_states = []\n        for layer_idx"),
    ("            if layer_idx in self.layers_to_capture:","            if layer_idx in tap_layers:"),
    ("        if self.capture_aux_hidden_states:\n            hidden_states, aux_hidden_states = hidden_states","        if self.capture_aux_hidden_states:\n            if getattr(forward_batch, \"dflash_skip_prefill_aux\", False):\n                if isinstance(hidden_states, tuple):\n                    # A replay of a statically captured body may still provide taps.\n                    hidden_states = hidden_states[0]\n            else:\n                hidden_states, aux_hidden_states = hidden_states")])
worker="python/sglang/srt/speculative/dflash_worker_v2.py"
start="            # Target prefill: capture DFlash aux hidden states for prompt tokens."
cond='''            coverage = getattr(self, "_window_coverage", None)
            if (coverage is not None and batch.forward_mode == ForwardMode.EXTEND
                and not batch.is_extend_in_batch
                and coverage.prepare(batch.reqs, batch.prefix_lens, batch.extend_lens)):'''
return_='''                logits_output.hidden_states = None
                batch_output.next_draft_input = self._make_next_draft_input_prefill(
                    bonus_tokens=next_token_ids, seq_lens=new_seq_lens)
                return batch_output'''
init="        self._window_coverage = DFlashRegionCoverage(window, page_size)"
modify(worker,[(init,'''        from sglang.srt.model_executor.forward_batch_info import get_server_return_hidden_states_mode

        target = self.target_worker.model_runner
        self._can_omit_target_aux = (
            any(getattr(m, "supports_dflash_window_aux_omission", False) for m in target.model.modules())
            and not get_server_return_hidden_states_mode().need_capture()
            and not any(getattr(m, "keep_last_hidden_states", False) or getattr(m, "expected_aux_hidden_states", None) is not None
                for m in target.model.modules())
        )
'''+init),(cond+"\n"+return_+"\n\n",""),
    (start,'''            coverage = getattr(self, "_window_coverage", None)
            omit = (coverage is not None and batch.forward_mode == ForwardMode.EXTEND
                and not batch.is_extend_in_batch
                and coverage.prepare(batch.reqs, batch.prefix_lens, batch.extend_lens))
            skip_aux = (omit and self._can_omit_target_aux
                and not batch.return_hidden_states_mode.need_capture())
'''+start),
    ("                capture_hidden_mode=CaptureHiddenMode.FULL,\n                return_kv_loc_plan=True,","                capture_hidden_mode=CaptureHiddenMode.FULL,\n                dflash_skip_prefill_aux=bool(skip_aux),\n                return_kv_loc_plan=True,"),
    ("            if logits_output.hidden_states is None:\n                raise RuntimeError(\n                    \"DFLASH requires target aux","            if omit:\n"+return_+"\n\n            if logits_output.hidden_states is None:\n                raise RuntimeError(\n                    \"DFLASH requires target aux")])
out=base/"early_candidate.diff";out.write_bytes(git("diff","--cached","--no-ext-diff","--binary",pin))
git("read-tree",pin);git("apply","--cached","--check",str(out))
assert max(len(x[1:]) for x in out.read_bytes().splitlines() if x.startswith(b'+'))<=400
print(len(out.read_bytes()),hashlib.sha256(out.read_bytes()).hexdigest())

