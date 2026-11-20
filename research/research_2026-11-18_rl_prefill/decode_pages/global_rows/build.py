"""Extend the verified window-only patch with pinned FA3 physical request-row indirection."""
import ast,hashlib,importlib.util,re,subprocess,tempfile
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent;ROOT=P.parents[1]
spec=importlib.util.spec_from_file_location('pages_base',P/'build.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
B,K,rep=base.B,base.K,base.replace
def change(path,s):
    if path==B:
        s=rep(s,'from sglang.srt.layers.attention.base_attn_backend import AttentionBackend','from sglang.srt.layers.attention.base_attn_backend import AttentionBackend, SharedReadEnds')
        s=rep(s,'    scheduler_metadata: torch.Tensor = None','''    scheduler_metadata: torch.Tensor = None
    muse_full_batch_indices: torch.Tensor = None''')
        s=rep(s,'    def _set_decode_page_metadata(','''    def shared_read_ends(self, fm):
        if self._muse_fa3_window_pages and (fm.is_decode() or fm.is_target_verify()):
            # Direct FA3 page lookup reads the scheduler-owned table across
            # decoder layers. Publish only after the last model read.
            return SharedReadEnds.POST_REPLAY
        return super().shared_read_ends(fm)

    def _set_decode_page_metadata(''')
        s=rep(s,'        normal_decode_set_metadata(\n            metadata.cache_seqlens_int32,\n            metadata.cu_seqlens_k,\n            metadata.page_table,\n            self.req_to_token,\n            req_pool_indices,','''        # A decode capture may later receive spec_info without a narrow
        # query bound. Keep its captured row-map pointer current; refresh the
        # entire SWA table in that case.
        if metadata.muse_full_batch_indices is not None and not window_query_length:
            window_query_length = self.max_num_pages
        eligible = (
            window_query_length and self._muse_fa3_window_pages
            and self.use_sliding_window_kv_pool and not translated and self.topk <= 1
            and self._kv_shard_pool is None and self.local_attn_builder is None
            and not self.is_prefill_aware_swa and self.sliding_window_size is not None
            and self.sliding_window_size > 0
        )
        direct = eligible and (
            self.req_to_token.dtype == torch.int32 and self.req_to_token.stride(1) == 1
            and self.req_to_token.shape[1] >= metadata.page_table.shape[1]
            and self.req_to_token.shape[0] < 2**31 and metadata.swa_page_table is not None
            and not self.is_encoder_decoder and self.attn_cp_size == 1
            and not self.kv_index_translator.is_translating
        )
        if direct and metadata.muse_full_batch_indices is None:
            metadata.muse_full_batch_indices = torch.empty_like(
                metadata.cache_seqlens_int32, dtype=torch.int32
            )
        normal_decode_set_metadata(
            metadata.cache_seqlens_int32,
            metadata.cu_seqlens_k,
            metadata.page_table,
            self.req_to_token,
            req_pool_indices,''')
        s=rep(s,'''                if window_query_length and self._muse_fa3_window_pages
                and self.use_sliding_window_kv_pool and not translated
                and self.topk <= 1 and self._kv_shard_pool is None
                and self.local_attn_builder is None and not self.is_prefill_aware_swa
                and self.sliding_window_size is not None and self.sliding_window_size > 0
                else 0
            ),''','''                if eligible else 0
            ),
            direct_full_indices=metadata.muse_full_batch_indices if direct else None,''')
        s=rep(s,'window_query_length=verify_delta if ragged_layout is None else 0,','''window_query_length=(
                        verify_delta if ragged_layout is None else
                        (ragged_layout.cap or ragged_layout.graph_num_tokens)
                    ),''')
        # The tested standard Muse branches first resolve the exact per-layer table, then optionally bind the full row map.
        for pre in ('                result = flash_attn_with_kvcache(\n                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),\n                    k_cache=key_cache,\n                    v_cache=value_cache,\n                    page_table=page_table,\n                    cache_seqlens=cache_seqlens,\n                    cu_seqlens_q=cu_seqlens_q,\n                    cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,\n                    scheduler_metadata=sched_meta,', '                result = flash_attn_with_kvcache(\n                    q=q_reshaped,\n                    k_cache=key_cache,\n                    v_cache=value_cache,\n                    page_table=page_table,\n                    cache_seqlens=cache_seqlens,\n                    cu_seqlens_q=metadata.cu_seqlens_q,'):
            s=rep(s,pre,'''                if (metadata.muse_full_batch_indices is not None
                        and not is_swa_layer and page_table is metadata.page_table):
                    kwargs["cache_batch_idx"] = metadata.muse_full_batch_indices
                    # Preserve the original capacity used by the captured scheduler.
                    page_table = self.req_to_token[:, :metadata.page_table.shape[1]]
'''+pre)
        return s
    s=rep(s,'    SWA_SUFFIX_TOKENS: tl.constexpr = 0,','    SWA_SUFFIX_TOKENS: tl.constexpr = 0,\n    DIRECT_FULL_INDICES=None,')
    s=rep(s,'            tl.store(cache_seqlens_int32 + idx * cache_seqlens_int32_stride_0, val)','''            if DIRECT_FULL_INDICES is not None:
                physical_row = tl.load(req_pool_indices + idx * req_pool_indices_stride_0)
                tl.store(DIRECT_FULL_INDICES + idx, physical_row.to(tl.int32))
            tl.store(cache_seqlens_int32 + idx * cache_seqlens_int32_stride_0, val)''')
    s=rep(s,'    col_start = pid_c * BLOCK_COLS\n    if not ZERO_TAIL', '''    col_start = pid_c * BLOCK_COLS
    if DIRECT_FULL_INDICES is not None:
        # Address the last complete boundary CTA and all following CTAs.
        first_token = tl.maximum(0, seq_len + SEQ_LEN_DELTA - SWA_SUFFIX_TOKENS)
        col_start += (first_token // (PAGE_SIZE * BLOCK_COLS)) * BLOCK_COLS
    if not ZERO_TAIL''')
    a=s.index('    full_pages = _resolve_metadata_page_ids(');b=s.index('    if swa_page_table is not None:',a)
    s=s[:a]+'    if DIRECT_FULL_INDICES is None:\n'+''.join('    '+line for line in s[a:b].splitlines(True))+s[b:]
    s=rep(s,'    swa_suffix_tokens: int = 0,\n):','    swa_suffix_tokens: int = 0,\n    direct_full_indices: Optional[torch.Tensor] = None,\n):')
    s=rep(s,'    _fused_decode_metadata_kernel[grid](','''    if direct_full_indices is not None:
        assert page_size == 1 and swa_suffix_tokens and direct_full_indices.dtype == torch.int32
        assert direct_full_indices.is_contiguous() and direct_full_indices.numel() == batch_size
        grid = (batch_size, min(grid[1], triton.cdiv(swa_suffix_tokens, block_cols) + 1))
    _fused_decode_metadata_kernel[grid](''')
    return rep(s,'        SWA_SUFFIX_TOKENS=swa_suffix_tokens,','        SWA_SUFFIX_TOKENS=swa_suffix_tokens,\n        DIRECT_FULL_INDICES=direct_full_indices,')
def main():
    import os
    assert hashlib.sha256((P/'candidate.diff').read_bytes()).hexdigest()=='1fcc5661c9896094f917d4e7d08c10c014d62fc3e5bf1bceb4fbf83bcbe5844c'
    env=dict(os.environ,GIT_INDEX_FILE=str(H/'global.index'));git=lambda *a,**kw:subprocess.check_output(['git','-C',str(base.REPO),*a],env=env,**kw)
    import difflib
    git('read-tree',base.PIN);git('apply','--cached',str(P/'candidate.diff'))
    for p in (K,B):
        old=git('show',':'+p).decode();new=change(p,old);ast.parse(new)
        patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),'a/'+p,'b/'+p)).encode();git('apply','--cached','-',input=patch)
        (H/Path(p).name).write_bytes(new.encode())
    dest=H/'candidate.diff';dest.write_bytes(git('diff','--cached','--binary',base.PIN))
    git('read-tree',base.PIN);git('apply','--cached','--check',str(dest));print(len(dest.read_bytes()),hashlib.sha256(dest.read_bytes()).hexdigest())
if __name__=='__main__':main()

