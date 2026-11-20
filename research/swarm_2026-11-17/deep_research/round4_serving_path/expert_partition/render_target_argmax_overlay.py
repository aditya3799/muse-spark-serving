import ast, difflib, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3].parent.parent
# Locate repository by the pinned checkout rather than relying on working directory.
ROOT=next(p for p in HERE.parents if (p/'sglang/python/sglang/srt').is_dir())
SRC=ROOT/'sglang/python'
OUT=HERE/'target_argmax_overlay/python'
rows=[]; diffs=[]
def sha(b): return hashlib.sha256(b).hexdigest()
def emit(rel, text, original=None):
    ast.parse(text)
    dest=OUT/rel; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(text,encoding='utf-8')
    rows.append({'path':rel,'source_lf_normalized_sha256':None if original is None else sha(original.encode()),'overlay_lf_normalized_sha256':sha(text.encode()),'overlay_file_sha256':sha(dest.read_bytes())})
    diffs.extend(difflib.unified_diff((original or '').splitlines(True),text.splitlines(True),fromfile='a/python/'+rel if original is not None else '/dev/null',tofile='b/python/'+rel))
def patch(rel, edits):
    original=(SRC/rel).read_text(encoding='utf-8-sig'); text=original
    for old,new in edits:
        assert text.count(old)==1,(rel,old,text.count(old)); text=text.replace(old,new)
    reverse=text
    for old,new in reversed(edits):
        assert reverse.count(new)==1; reverse=reverse.replace(new,old)
    assert reverse==original
    emit(rel,text,original)
base='sglang/srt/'
patch(base+'speculative/dflash_worker_v2.py',[
 ('        self._target_logits_processor = self._defer_target_verify_gather()\n',
  '        self._target_logits_processor = self._defer_target_verify_gather()\n        from sglang.srt.speculative.dflash_target_argmax_adapter import maybe_install_target_argmax_capture\n        self._research_target_argmax_capture = maybe_install_target_argmax_capture(self)\n'),
 ('                target_predict = target_lp.deferred_argmax(\n                    logits_output.next_token_logits\n                )\n',
  '                capture_adapter = self._research_target_argmax_capture\n                if capture_adapter is not None:\n                    target_predict = capture_adapter.consume(\n                        raw_rows=bs * int(self.block_size),\n                        graph_used=can_run_cuda_graph,\n                        can_skip_verify_logits=True,\n                    )\n                if target_predict is None:\n                    target_predict = target_lp.deferred_argmax(\n                        logits_output.next_token_logits\n                    )\n')])
patch(base+'model_executor/runner/decode_cuda_graph_runner.py',[
 ('    def capture(self) -> None:\n        with self.model_context():\n            self._capture()\n',
  '    def capture(self) -> None:\n        with self.model_context():\n            try:\n                for hook in self.model_runner.capture_tail_hooks:\n                    notify = getattr(hook, "on_capture_start", None)\n                    if notify is not None:\n                        notify(self)\n                self._capture()\n            except BaseException:\n                for hook in self.model_runner.capture_tail_hooks:\n                    notify = getattr(hook, "on_capture_abort", None)\n                    if notify is not None:\n                        notify(self)\n                raise\n'),
 ('                self.backend.capture_one(\n                    shape_key,\n                    run_once,\n                    capture_inputs=None,\n                    post_warmup_hook=post_warmup_hook,\n                )\n',
  '                self.backend.capture_one(\n                    shape_key,\n                    run_once,\n                    capture_inputs=None,\n                    post_warmup_hook=post_warmup_hook,\n                )\n                for hook in self.model_runner.capture_tail_hooks:\n                    notify = getattr(hook, "on_capture_success", None)\n                    if notify is not None:\n                        notify(self, forward_batch, num_tokens, shape_key)\n')])
core=HERE.parent/'coordination/target_argmax_capture.py'
emit(base+'speculative/dflash_target_argmax_capture.py',core.read_text(encoding='utf-8-sig'))
emit(base+'speculative/dflash_target_argmax_adapter.py',(HERE/'dflash_target_argmax_adapter.py').read_text(encoding='utf-8-sig'))
(HERE/'target_argmax_overlay.diff').write_text(''.join(diffs),encoding='utf-8')
report={'status':'INERT_DEFAULT_OFF_CPU_SOURCE_ONLY','files':rows,'core_source_sha256':sha(core.read_bytes()),'edits_reversible':True,'enable':'Explicitly set RESEARCH_ENABLE_TARGET_ARGMAX_CAPTURE=True in isolated adapter, identically on all ranks, before constructing workers. No environment or active checkout edits made.','scope':'TP8 CUDA DFlash8 greedy exact target graph buckets, FullCudaGraphBackend, serial same-stream only. Non-greedy graph calls pay extra captured argmax but retain original gather/sampling.','held':['real model startup and real NCCL graph capture','native multi-rank equality','serving performance','combination with prefix overlay','concurrent stream execution']}
(HERE/'TARGET_ARGMAX_OVERLAY_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
