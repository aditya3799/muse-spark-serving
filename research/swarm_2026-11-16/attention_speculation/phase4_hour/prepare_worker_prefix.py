"""Create isolated eager worker prototype; never edit active checkout."""
import ast,difflib,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
rel=Path('sglang/python/sglang/srt/speculative/dflash_worker_v2.py')
source=(ROOT/rel).read_text(encoding='utf-8')
text=source
replacements=[]
def replace(old,new):
    global text
    assert text.count(old)==1,(old[:80],text.count(old))
    text=text.replace(old,new);replacements.append((old,new))
replace('import logging\n','import logging\nfrom sglang.srt.speculative.dflash_prefix_research import resolve_verify_prefix\n')
replace('target_predict = target_predict.view(bs, int(self.block_size))',
        'target_predict = target_predict.view(bs, candidates.shape[1])')
replace('        return accept_len, commit_lens, bonus, out_tokens, new_seq_lens\n',
'''        # Preserve scheduler output stride8 even when the stock eager fallback
        # produces a compact [B,K] output instead of the persistent [B,8] buffer.
        if out_tokens.shape[1] != int(self.block_size):
            padded = out_tokens.new_zeros((bs, int(self.block_size)))
            padded[:, :out_tokens.shape[1]].copy_(out_tokens)
            out_tokens = padded
        return accept_len, commit_lens, bonus, out_tokens, new_seq_lens
''')
replace('        verify_input_ids = draft_tokens.reshape(-1)\n',
'''        # Research-only: full draft8 computation and allocation above are unchanged.
        # The default helper width is8. Prefix experiments are conservative greedy,
        # non-DP, non-Mamba, no grammar/logprobs/penalties/route-output paths.
        prefix_eligible = (
            self._can_skip_verify_logits(batch, batch.sampling_info)
            and not self._need_mamba_verify_commit
            and not get_parallel().attn_dp_enabled
            and SIMULATE_ACC_LEN == 0
            and batch.seq_lens_cpu is not None
            and not any(getattr(req, 'return_routed_experts', False) for req in batch.reqs)
            and getattr(batch, 'replace_embeds', None) is None
        )
        verify_width, verify_candidates, verify_positions, verify_slots, verify_cols = resolve_verify_prefix(
            draft_tokens, positions, verify_out_cache_loc, eligible=prefix_eligible
        )
        verify_input_ids = verify_candidates.reshape(-1)
''')
replace('''            draft_token=verify_input_ids,
            positions=positions,
            draft_token_num=int(self.block_size),
            custom_mask=custom_mask,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            kv_loc_plan=kv_loc_plan,
''','''            draft_token=verify_input_ids,
            positions=verify_positions,
            draft_token_num=verify_width,
            custom_mask=custom_mask,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            kv_loc_plan=kv_loc_plan,
            kv_loc_cols=verify_cols,
''')
replace('        batch.out_cache_loc = verify_out_cache_loc\n        sampling_info = batch.sampling_info',
        '        batch.out_cache_loc = verify_slots\n        sampling_info = batch.sampling_info')
replace('verify_host_seq_lens = seq_lens_cpu_backup + block_size','verify_host_seq_lens = seq_lens_cpu_backup + verify_width')
replace('''                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=int(self.block_size),
''','''                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=verify_width,
''')
replace('        candidates = draft_tokens\n','        candidates = verify_candidates\n')
replace('''            draft_input=draft_input,
            draft_token_num=int(self.block_size),
            bs=bs,
''','''            draft_input=draft_input,
            draft_token_num=verify_width,
            bs=bs,
''')
replace('hidden = hidden.view(bs, int(self.block_size), -1)','hidden = hidden.view(bs, verify_width, -1)')
replace('''        cache_loc = self._draft_write_ids(kv_loc_plan)
        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=cache_loc,
            cache_loc_2d=cache_loc.view(verify_out_cache_loc_2d.shape),
            positions=positions,
''','''        cache_loc = kv_loc_plan.write_ids(
            self.draft_model_runner.kv_index_translator, cols=verify_cols
        )
        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=cache_loc,
            cache_loc_2d=cache_loc.view(bs, verify_width),
            positions=verify_positions,
''')
# All source edits reversible; exact AST equality to original after reverse edits.
reverse=text
for old,new in reversed(replacements):
    assert reverse.count(new)==1
    reverse=reverse.replace(new,old)
assert reverse==source
ast.parse(text)
# Full draft generation before target boundary is byte-for-byte unchanged.
marker='        # --- 2) Target verify.'
assert source[source.index('        # --- 1) Draft a fixed block'):source.index(marker)]==text[text.index('        # --- 1) Draft a fixed block'):text.index(marker)]
out=P/'candidate_sources'/rel;out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(text,encoding='utf-8')
helper=out.parent/'dflash_prefix_research.py';helper.write_bytes((P/'dflash_prefix_research.py').read_bytes())
patch=''.join(difflib.unified_diff(source.splitlines(True),text.splitlines(True),fromfile='a/'+str(rel).replace('\\','/'),tofile='b/'+str(rel).replace('\\','/')))
(P/'worker_prefix_experiment.diff').write_text(patch,encoding='utf-8')
sha=lambda b:hashlib.sha256(b).hexdigest()
report={'scope':'Experimental eager source integration, default width8; not packaged, promoted or model tested.',
        'original_worker_sha256':sha((ROOT/rel).read_bytes()),'candidate_worker_sha256':sha(out.read_bytes()),
        'helper_sha256':sha(helper.read_bytes()),'reversible_source_edits':len(replacements),
        'full_draft_generation_unchanged':True,'reversed_AST_identical':True,
        'output_allocation_stride':8,'target_prefix_width_default':8,
        'graph_contract':'Existing capturewidth8 runner rejects K!=8 and falls back eager. Separate K graph required before a performance conclusion.'}
(P/'source_integration_manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
