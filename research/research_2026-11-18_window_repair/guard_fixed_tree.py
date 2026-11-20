from pathlib import Path
H=Path(__file__).resolve().parent
p=H/"tree_dispatch.py";s=p.read_text(encoding="utf8")
if "def tree_runtime_compatible(" not in s:
    start=s.index("    runner = worker.model_runner");end=s.index("    translator = runner.kv_index_translator",start)
    s=s[:start]+'''    if not tree_runtime_compatible(worker):
        return None
    runner=worker.model_runner
    backend=runner.attn_backend
    if (batch.forward_mode.is_idle() or not worker._can_skip_verify_logits(batch,batch.sampling_info)
        or (batch.sampling_info is not None and not batch.sampling_info.is_all_greedy)):
        return None
'''+s[end:]
    start=s.index("def select_tree(")
    s=s[:start]+'''def tree_runtime_compatible(worker):
    runner=worker.model_runner
    backend=runner.attn_backend
    par=get_parallel()
    return (worker.lilicorr is None and not worker._need_mamba_verify_commit
        and not par.attn_dp_enabled and par.attn_dcp_size==1 and not par.enable_prefill_cp
        and type(backend).__name__=="FlashAttentionBackend"
        and type(runner.model).__name__ in ("MuseSparkTextForCausalLM","MuseSparkForConditionalGeneration")
        and backend.fa_impl_ver==3 and backend.page_size==1 and not backend.use_mla
        and not backend.fa_skip_kv_cache and not backend.kv_cache_is_mxfp8 and not backend.has_local_attention
        and backend.use_sliding_window_kv_pool and backend.sliding_window_size>0
        and getattr(backend,"_kv_shard_pool",None) is None
        and getattr(runner.token_to_kv_pool,"swa_req_ring_size",None) is None)


'''+s[start:]
    s=s.replace("and backend.sliding_window_size>0","and (backend.sliding_window_size or -1)>0")
    p.write_text(s,encoding="utf8")
p=H/"build_fixed_tree.py";s=p.read_text(encoding="utf8")
key='old=rep(old,"        self.uniforms ='
if 'tree_enabled=tree_runtime_compatible' not in s:
    pre='''old=rep(old,"self, *, draft_model, block_size, max_bs, device, sampling_enabled: bool",
            "self, *, draft_model, block_size, max_bs, device, sampling_enabled: bool, tree_enabled=False")
old=rep(old,"            return _SelectorDraftSampler(", "            from sglang.srt.speculative.fixed_tree_dispatch import tree_runtime_compatible\\n            return _SelectorDraftSampler(\\n                tree_enabled=tree_runtime_compatible(self),")
'''
    s=s.replace(key,pre+key,1).replace("if gamma==7 and top_k==16 else None","if tree_enabled and gamma==7 and top_k==16 else None")
    p.write_text(s,encoding="utf8")
p=H/"test_tree_dispatch.py";s=p.read_text()
s=s.replace('(backend,"has_local_attention",True),','(backend,"_kv_shard_pool",object()),(runner.token_to_kv_pool,"swa_req_ring_size",2048),(backend,"has_local_attention",True),')
s=s.replace('checks=0','backend._kv_shard_pool=None;runner.token_to_kv_pool.swa_req_ring_size=None\nchecks=0')
p.write_text(s)

