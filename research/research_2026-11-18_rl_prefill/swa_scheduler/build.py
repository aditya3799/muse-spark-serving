"""Build a cumulative scheduler patch from the immutable scored candidate."""
import ast
import importlib.util
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
helper = importlib.util.spec_from_file_location("parent_builder", HERE.parent / "sequence_parallel" / "build.py")
base = importlib.util.module_from_spec(helper)
helper.loader.exec_module(base)
path = base.PREFIX + "layers/attention/flashattention_backend.py"
replace = base.replace

def edit(src):
    src = replace(src, "    get_exec,\n", "    get_disagg,\n    get_exec,\n")
    src = replace(src, "from sglang.srt.speculative.ragged_verify import build_ragged_target_verify_geometry",
                  "from sglang.srt.speculative.ragged_verify import (\n    build_ragged_target_verify_geometry,\n    ragged_verify_compact_enabled,\n)")
    src = replace(src, "    scheduler_metadata: torch.Tensor = None\n",
                  "    scheduler_metadata: torch.Tensor = None\n    swa_verify_scheduler: tuple = None\n")
    needle = "            _should_disable_scheduler_metadata_precompute()\n        )\n"
    src = replace(src, needle, needle + """        self._swa_verify_enabled = (
            self.fa_impl_ver == 3 and self._get_scheduler_metadata is not None
            and self.has_swa and self.sliding_window_size >= 0 and not self.use_mla
            and not self.is_draft_runner and self.topk <= 1 and not self.is_prefill_aware_swa
            and self.local_attn_builder is None and self._kv_shard_pool is None
            and not self._disable_scheduler_metadata_precompute and self.attn_cp_size == 1
            and not get_exec().overlap.enable_two_batch_overlap and not get_disagg().enable_pdmux
            and not get_exec().graph.enable_torch_compile and not ragged_verify_compact_enabled()
            and model_runner.spec_algorithm.is_dflash_family()
        )
""")
    src = replace(src, "    def validate_elastic_cuda_graph_recapture(", (HERE / "methods.py.inc").read_text(encoding="utf-8") + "\n    def validate_elastic_cuda_graph_recapture(")
    needle = "    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:\n        if self._capture_unified_verify_metadata(forward_batch):"
    src = replace(src, needle, """    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:\n        if self._swa_verify_enabled and forward_batch.forward_mode.is_target_verify():
            self._prepare_swa_verify_scheduler(self.forward_metadata, forward_batch)
        if self._capture_unified_verify_metadata(forward_batch):
""")
    needle = "                metadata, batch_size\n            )\n        self.forward_metadata = metadata"
    src = replace(src, needle, """                metadata, batch_size
            )
            if self._swa_verify_enabled:
                self._prepare_swa_verify_scheduler(metadata, forward_batch)
        self.forward_metadata = metadata""")
    needle = "                    sched_meta = metadata.scheduler_metadata\n                result = flash_attn_with_kvcache(\n"
    return replace(src, needle, """                    sched_meta = metadata.scheduler_metadata
                elif is_swa_layer and metadata.swa_verify_scheduler is not None:
                    sched_meta = self._swa_verify_scheduler_for_layer(
                        metadata, forward_batch, layer, q, key_cache, value_cache, page_table,
                        cache_seqlens, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, window_size,
                        causal, cp_active, use_local_attn, use_cascade_attn, sinks, score_mod, rel_bias,
                    )
                result = flash_attn_with_kvcache(
""")

def main():
    original = (base.ROOT / "release_2026-11-18/candidate.diff").read_bytes()
    assert hashlib.sha256(original).hexdigest() == base.SCORED
    pieces = re.split(r"(?=^diff --git )", original.decode(), flags=re.M)[1:]
    sections = {piece.splitlines()[0].split()[2][2:]:piece for piece in pieces}
    pristine = subprocess.check_output(["git", "-C", str(base.REPO), "show", base.BASE+":"+path]).decode()
    parent = (HERE.parent / "decode_qkv" / "parent_flashattention_backend.py").read_text(encoding="utf-8")
    edited = edit(parent)
    ast.parse(edited)
    delta = base.diff(path, parent, edited).encode()
    sections[path] = base.diff(path, pristine, edited)
    candidate = "".join(sections.values()).encode()
    with tempfile.TemporaryDirectory(dir=HERE) as temp:
        def run(args, **kw):
            return subprocess.run(["git", "-C", temp, *args], check=True, **kw)
        run(["init", "-q"])
        for name in sections:
            a = subprocess.run(["git","-C",str(base.REPO),"show",base.BASE+":"+name],capture_output=True)
            if a.returncode == 0:
                dest = Path(temp)/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(a.stdout)
        run(["apply","--check","-"], input=candidate)
        run(["apply","-"], input=original)
        assert (Path(temp)/path).read_text(encoding="utf-8") == parent
        run(["apply","--check","-"], input=delta)
        run(["apply","-"], input=delta)
        assert (Path(temp)/path).read_text(encoding="utf-8") == edited
    for name,data in (("candidate.diff",candidate),("delta.diff",delta),("flashattention_backend.py",edited.encode())):
        (HERE / name).write_bytes(data)
    result=dict(base=base.BASE,scored=base.SCORED,sha256=hashlib.sha256(candidate).hexdigest(),
                bytes=len(candidate),pristine_apply=True,scored_delta_apply=True)
    (HERE/"build.json").write_text(json.dumps(result,indent=2))
    print(result)

if __name__ == "__main__":
    main()
