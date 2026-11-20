"""Build a window-aware FA3 decode metadata writer for the scored Muse parent."""
import ast,hashlib,re,subprocess,tempfile
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[1];REPO=ROOT/'sglang'
K='python/sglang/kernels/ops/attention/metadata.py'
B='python/sglang/srt/layers/attention/flashattention_backend.py'
PIN='297e54fd105e974cdf401e5b4d20253a22c16f90'
S='8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b'
def replace(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b)
def edit(p,s):
    if p==B:
        s=replace(s,"        # Static page-table width (upper bound).",'''        # The pinned Hopper FA3 local mainloop rebases paged reads to at
        # least seqlen_k - seqlen_q - window_left. Muse has no sink tokens.
        self._muse_fa3_window_pages = (
            fa_impl_ver == 3 and not model_runner.is_draft_worker
            and getattr(model_runner.model_config.hf_config, "model_type", "") in
                ("muse_spark", "muse_spark_text")
            and self.kv_cache_dtype == torch.float8_e4m3fn and self.page_size == 1
            and torch.cuda.get_device_capability(self.device)[0] == 9
        )
        # Static page-table width (upper bound).''')
        s=replace(s,"        seq_len_delta: int,\n    ) -> None:\n        \"\"\"Fill",'''        seq_len_delta: int,
        *,
        window_query_length: int = 0,
    ) -> None:
        """Fill''')
        s=replace(s,"            skip_page_table=translated,\n        )",'''            skip_page_table=translated,
            swa_suffix_tokens=(
                window_query_length + self.sliding_window_size
                if window_query_length and self._muse_fa3_window_pages
                and self.use_sliding_window_kv_pool and not translated
                and self.topk <= 1 and self._kv_shard_pool is None
                and self.local_attn_builder is None and not self.is_prefill_aware_swa
                and self.sliding_window_size is not None and self.sliding_window_size > 0
                else 0
            ),
        )''')
        s=replace(s,"                        self._spec_read_seq_len_delta(forward_mode, spec_info),\n                    )\n\n                if self.local_attn_builder",'''                        self._spec_read_seq_len_delta(forward_mode, spec_info),
                        window_query_length=1,
                    )

                if self.local_attn_builder''')
        return replace(s,"                    plan, metadata, req_pool_indices, verify_lens, verify_delta\n                )",'''                    plan, metadata, req_pool_indices, verify_lens, verify_delta,
                    window_query_length=verify_delta if ragged_layout is None else 0,
                )''')
    s=replace(s,"    ZERO_TAIL: tl.constexpr,\n):\n    pid_b", "    ZERO_TAIL: tl.constexpr,\n    SWA_SUFFIX_TOKENS: tl.constexpr = 0,\n):\n    pid_b")
    s=replace(s,"    if swa_page_table is not None:\n        swa_pages =",'''    if swa_page_table is not None:
        if SWA_SUFFIX_TOKENS:
            # Keep a complete boundary CTA. Every full-attention page and all
            # sequence lengths are still rewritten before reaching this branch.
            first_token = seq_len + SEQ_LEN_DELTA - SWA_SUFFIX_TOKENS
            if (col_start + BLOCK_COLS) * PAGE_SIZE <= first_token:
                return
        swa_pages =''')
    s=replace(s,"    zero_tail: bool = False,\n):","    zero_tail: bool = False,\n    swa_suffix_tokens: int = 0,\n):")
    s=replace(s,"    batch_size = cache_seqlens_int32.shape[0]\n    # Lengths-only",'''    if swa_suffix_tokens:
        assert swa_suffix_tokens > 0 and not zero_tail and not skip_page_table
        assert swa_page_table is not None and full_to_swa_index_mapping is not None
    batch_size = cache_seqlens_int32.shape[0]
    # Lengths-only''')
    s=replace(s,"        if skip_page_table\n        or (page_size", "        if skip_page_table or swa_suffix_tokens\n        or (page_size")
    return replace(s,"        ZERO_TAIL=zero_tail,","        ZERO_TAIL=zero_tail,\n        SWA_SUFFIX_TOKENS=swa_suffix_tokens,")
def main():
    import difflib
    base=(ROOT/'release_2026-11-18/candidate.diff').read_bytes();assert hashlib.sha256(base).hexdigest()==S
    chunks=re.split(r"(?=^diff --git )",base.decode(),flags=re.M)[1:]
    parts={c.splitlines()[0].split()[2][2:]:c for c in chunks}
    def diff(p,a,b):return f'diff --git a/{p} b/{p}\n'+''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),'a/'+p,'b/'+p))
    with tempfile.TemporaryDirectory(dir=H) as t:
        subprocess.run(['git','-C',t,'init','-q'],check=True);changes=[]
        for p in (B,K):
            raw=subprocess.check_output(['git','-C',str(REPO),'show',PIN+':'+p]).decode()
            d=Path(t)/p;d.parent.mkdir(exist_ok=True,parents=True);d.write_bytes(raw.encode())
            if p in parts:subprocess.run(['git','-C',t,'apply','-'],input=parts[p].encode(),check=True)
            old=d.read_bytes().decode();new=edit(p,old);ast.parse(new)
            (H/('parent_'+Path(p).name)).write_bytes(old.encode());(H/Path(p).name).write_bytes(new.encode())
            changes.append(diff(p,old,new));parts[p]=diff(p,raw,new)
        inc=''.join(changes).encode();subprocess.run(['git','-C',t,'apply','--check','-'],input=inc,check=True)
        (H/'delta.diff').write_bytes(inc);body=''.join(parts.values()).encode();(H/'candidate.diff').write_bytes(body)
        print(len(body),hashlib.sha256(body).hexdigest())
if __name__=='__main__':main()

