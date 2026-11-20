from pathlib import Path
from difflib import unified_diff
import ast,hashlib,os,subprocess
base=Path(__file__).parent.resolve()
root=base.parent
repo=root/"sglang"
pin="297e54fd105e974cdf401e5b4d20253a22c16f90"
parent=(root/"release_2026-11-18/candidate.diff").read_bytes()
assert hashlib.sha256(parent).hexdigest()=="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
pieces=[]
def stage(name,replacements=None,new=None):
    old=subprocess.check_output(["git","-C",str(repo),"show",pin+":"+name]).decode() if replacements else ""
    assert ("diff --git a/"+name+" b/"+name).encode() not in parent
    if replacements:
        new=old
        for a,b in replacements:
            assert new.count(a)==1,a
            new=new.replace(a,b)
    ast.parse(new)
    pieces.append("diff --git a/"+name+" b/"+name+"\n"+("new file mode 100644\n" if not old else "")+"".join(unified_diff(old.splitlines(True),new.splitlines(True),"a/"+name if old else "/dev/null","b/"+name)))
    f=base/"applied_regions"/name
    f.parent.mkdir(parents=True,exist_ok=True)
    f.write_text(new)
stage("python/sglang/srt/speculative/dflash_window_coverage.py",new=(base/"regions.py").read_text())
start="        self.tree_cache = result.tree_cache\n"
stage("python/sglang/srt/managers/scheduler.py",[(start,start+'''        builder = getattr(self.draft_worker, "build_window_coverage", None)
        if (builder is not None and get_disagg().disaggregation_mode == "null"
            and not self.enable_hicache_storage and get_parallel().pp_size == 1
            and not self.tree_cache.supports_mamba()):
            self.tree_cache._draft_window_coverage = builder(self.page_size)
''')])
start="    def init_cuda_graphs(self):\n"
hook="            # Materialize prompt tokens into the draft KV cache immediately."
stage("python/sglang/srt/speculative/dflash_worker_v2.py",[(start,'''    def build_window_coverage(self, page_size):
        from sglang.srt.speculative.dflash_window_coverage import DFlashRegionCoverage

        if (self._is_domino or self.lilicorr is not None or get_parallel().attn_dp_enabled
            or not is_cuda()):
            return None
        config = self.draft_model.config
        layers = getattr(config, "layer_types", None)
        window = getattr(config, "sliding_window", None)
        if (not layers or not all(kind == "sliding_attention" for kind in layers)
            or len(layers) != len(self.draft_model.layers) or not window or window <= 0):
            return None
        self._window_coverage = DFlashRegionCoverage(window, page_size)
        return self._window_coverage

'''+start),(hook,'''            coverage = getattr(self, "_window_coverage", None)
            if (coverage is not None and batch.forward_mode == ForwardMode.EXTEND
                and not batch.is_extend_in_batch
                and coverage.prepare(batch.reqs, batch.prefix_lens, batch.extend_lens)):
                logits_output.hidden_states = None
                batch_output.next_draft_input = self._make_next_draft_input_prefill(
                    bonus_tokens=next_token_ids, seq_lens=new_seq_lens)
                return batch_output

'''+hook)])
start="    return RadixKey(\n        token_ids=token_ids,\n        extra_key=req.extra_key,\n        limit=key_limit,"
ck="    if not req.skip_radix_cache_insert:\n        tree_cache.checkpoint(req, up_to=req.extend_end)"
rel="    checkpoint = checkpoint and not req.skip_radix_cache_insert"
stage("python/sglang/srt/mem_cache/common.py",[(start,'''    coverage = getattr(tree_cache, "_draft_window_coverage", None)
    if coverage is not None:
        key_limit = coverage.match_limit(req, token_ids, key_limit)
'''+start),(ck,'''    coverage = getattr(tree_cache, "_draft_window_coverage", None)
    if not req.skip_radix_cache_insert and (coverage is None or coverage.may_publish(req, req.extend_end)):
        tree_cache.checkpoint(req, up_to=req.extend_end)'''),(rel,rel+'''
    coverage = getattr(tree_cache, "_draft_window_coverage", None)
    if checkpoint and coverage is not None:
        checkpoint = coverage.may_publish(req, owned_kv_len)''')])
out=base/"region_candidate.diff"
data=parent+"".join(pieces).encode()
assert max(len(x[1:]) for x in data.decode().splitlines() if x.startswith("+"))<=400
out.write_bytes(data)
env={**os.environ,"GIT_INDEX_FILE":str(base/"region.index")}
subprocess.run(["git","-C",str(repo),"read-tree",pin],env=env,check=True)
subprocess.run(["git","-C",str(repo),"apply","--cached","--check",str(out)],env=env,check=True)
print(len(data),hashlib.sha256(data).hexdigest())

