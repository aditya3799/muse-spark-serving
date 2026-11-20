from pathlib import Path
import ast, difflib, hashlib, json

out=Path(__file__).parent/'draft_glue'
root=Path(__file__).parent.parent.parent
rel='python/sglang/srt/layers/attention/flashattention_backend.py'
source=(root/'sglang'/rel).read_text(encoding='utf-8')
needle='    def validate_elastic_cuda_graph_recapture(self) -> None:'
addition='''    def glue_draft_metadata(self) -> str:
        # Only this fixed-width FA4 path reads runner slots and persistent
        # pool tensors exclusively. In particular, a translated KVLocPlan can
        # cache step-owned tables and must keep running its Python planner.
        if not (
            self.is_draft_runner
            and SpeculativeAlgorithm.from_string(
                get_spec().speculative_algorithm
            ).is_dflash()
            and self.fa_impl_ver == 4
            and self.topk <= 1
            and self.speculative_num_draft_tokens is not None
        ):
            return "not a fixed-width FA4 DFlash draft"
        if (
            self.is_encoder_decoder
            or self.use_mla
            or self.local_attn_builder is not None
            or self.use_sliding_window_kv_pool
            or self._kv_shard_pool is not None
            or self.kv_index_translator.is_translating
            or self.kv_index_translator.reads_are_translated
            or self.needs_cpu_seq_lens
            or self.is_prefill_aware_swa
            or get_parallel().attn_cp_size != 1
            or _should_disable_scheduler_metadata_precompute()
        ):
            return "draft metadata has a host-fed or translated layout"
        return ""

'''
assert source.count(needle)==1
candidate=source.replace(needle,addition+needle)
ast.parse(candidate)
dest=out/'candidate'/rel
dest.parent.mkdir(parents=True,exist_ok=True)
dest.write_bytes(candidate.encode())
diff=''.join(difflib.unified_diff(source.splitlines(True),candidate.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
(out/'draft_glue_delta.diff').write_bytes(diff.encode())
(out/'manifest.json').write_text(json.dumps(dict(status='experimental_metadata_only_not_promotable',candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),delta_bytes=len(diff.encode()),guard='DFLASH draft, FA4, static topk<=1, nontranslated non-SWA-pool non-MLA nonlocal noDP/CP nohostlengths',risk='Requires exact metadata replay and live mutable pool/request tests before candidate use; no acceptance/performance evidence'),indent=2))
print('Exported experimental draft glue',len(diff.encode()))
