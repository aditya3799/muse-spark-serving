from pathlib import Path
import ast, difflib, hashlib, json

out = Path(__file__).resolve().parent
root = out.parent.parent
rel = 'python/sglang/srt/layers/attention/flashattention_backend.py'
source = (root / 'sglang' / rel).read_text(encoding='utf-8')
new = source

def replace(old, replacement):
    global new
    assert new.count(old) == 1, (old[:80], new.count(old))
    new = new.replace(old, replacement)

replace('''        self, batch_size, max_seq_len_k, cache_seqlens, cu_seqlens_q
    ):
        """Compute FA3 scheduler metadata for decode.''', '''        self, batch_size, max_seq_len_k, cache_seqlens, cu_seqlens_q,
        max_seq_len_q=1,
    ):
        """Compute FA3 scheduler metadata for causal full attention.''')
replace('''            max_seqlen_q=1,
            max_seqlen_k=max_seq_len_k,''', '''            max_seqlen_q=max_seq_len_q,
            max_seqlen_k=max_seq_len_k,''')
replace('''    def validate_elastic_cuda_graph_recapture(self) -> None:''', '''    def _compute_verify_scheduler_metadata(self, metadata, batch_size):
        # Local and distributed attention can change the effective geometry.
        # Retain their per-layer scheduler path.
        if (
            self.local_attn_builder is not None
            or self._kv_shard_pool is not None
            or self.topk > 1
        ):
            return None
        return self._compute_scheduler_metadata(
            batch_size,
            # FA3 derives its maximum K extent from this same paged-table
            # capacity. A shorter live host bound can change metadata shape.
            metadata.page_table.shape[1] * self.page_size,
            metadata.cache_seqlens_int32,
            metadata.cu_seqlens_q,
            max_seq_len_q=metadata.max_seq_len_q,
        )

    def validate_elastic_cuda_graph_recapture(self) -> None:''')
replace('''        self.forward_metadata = metadata

    def get_paged_mha_kv_cache(''', '''        # The page table must already have been converted to pages, so
        # scheduler capacity matches the table passed to FA3 exactly.
        if forward_batch.forward_mode.is_target_verify() and self.topk <= 1:
            metadata.scheduler_metadata = self._compute_verify_scheduler_metadata(
                metadata, batch_size
            )
        self.forward_metadata = metadata

    def get_paged_mha_kv_cache(''')
replace('''        else:
            self._sched_meta_buf = None

        if self.local_attn_builder is not None:''', '''            # Verify has max_seqlen_q > 1 and needs four rounded vectors.
            # Keep its storage separate from decode's captured views.
            self._verify_sched_meta_buf = torch.zeros(
                1 + b_rounded * 4, dtype=torch.int32, device=self.device
            )
        else:
            self._sched_meta_buf = None
            self._verify_sched_meta_buf = None

        if self.local_attn_builder is not None:''')
replace('''            if forward_mode.is_draft_extend_v2():
                # CUDA graph bakes max_seq_len_q''', '''            if forward_mode.is_target_verify() and self.topk <= 1:
                metadata = self.target_verify_metadata[bs]
                if self._verify_sched_meta_buf is not None:
                    sched = self._compute_verify_scheduler_metadata(metadata, bs)
                    if sched is not None:
                        n = sched.numel()
                        assert n <= self._verify_sched_meta_buf.numel()
                        self._verify_sched_meta_buf[:n].copy_(sched)
                        self._verify_sched_meta_buf[n:].zero_()
                        metadata.scheduler_metadata = self._verify_sched_meta_buf[:n]

            if forward_mode.is_draft_extend_v2():
                # CUDA graph bakes max_seq_len_q''')
replace('''                # Page table built on-device (self-guards on cache_seqlens);
                # max_seq_len_k left unset -- unread here (scheduler_metadata is
                # normal-decode-only).
                self._set_decode_page_metadata(
                    plan, metadata, req_pool_indices, verify_lens, verify_delta
                )''', '''                # Both attention and its scheduler use the page-table capacity;
                # live lengths remain device data and need no host maximum.
                self._set_decode_page_metadata(
                    plan, metadata, req_pool_indices, verify_lens, verify_delta
                )
                if metadata.scheduler_metadata is not None:
                    sched = self._compute_verify_scheduler_metadata(metadata, bs)
                    if sched is not None:
                        n = sched.numel()
                        assert n == metadata.scheduler_metadata.numel()
                        self._verify_sched_meta_buf[:n].copy_(sched)
                        self._verify_sched_meta_buf[n:].zero_()''')
replace('''            else:
                result = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,''', '''            else:
                # Only reuse a scheduler for exactly the geometry it describes.
                # SWA, cross/local attention and custom masks retain stock paths.
                sched_meta = None
                if (
                    forward_batch.forward_mode.is_target_verify()
                    and self.topk <= 1
                    and metadata.scheduler_metadata is not None
                    and causal
                    and not is_swa_layer
                    and not use_local_attn
                    and not layer.is_cross_attention
                    and sinks is None
                    and score_mod is None
                    and rel_bias is None
                    and metadata.image_token_ranges is None
                    and q.dtype == self.kv_cache_dtype
                    and layer.tp_q_head_num == self.num_attention_heads
                    and layer.tp_k_head_num == self.num_kv_heads
                    and layer.head_dim == self.head_dim
                    and layer.v_head_dim == self.head_dim
                    and bool(layer.logit_cap > 0) == self.has_softcap
                ):
                    sched_meta = metadata.scheduler_metadata
                result = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                    scheduler_metadata=sched_meta,''')

ast.parse(new)
path = out / 'candidate' / rel
path.parent.mkdir(parents=True, exist_ok=True)
path.write_bytes(new.encode())
diff = ''.join(difflib.unified_diff(source.splitlines(True), new.splitlines(True), fromfile='a/'+rel, tofile='b/'+rel))
(out/'attention_delta.diff').write_bytes(diff.encode())
(out/'candidate_manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(), source_raw_sha256=hashlib.sha256((root/'sglang'/rel).read_bytes()).hexdigest(), encoding='utf-8', newline='LF', candidate_sha256=hashlib.sha256(new.encode()).hexdigest(), delta_bytes=len(diff.encode()), files=[rel], status='experimental_not_h200_validated'), indent=2))
print('Exported candidate', len(diff.encode()), 'bytes')
