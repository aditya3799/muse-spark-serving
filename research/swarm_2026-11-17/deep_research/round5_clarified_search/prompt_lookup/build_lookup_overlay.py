import ast,difflib,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[3]
f=R/'sglang/python/sglang/srt/speculative/dflash_worker_v2.py'
s=f.read_text(encoding='utf-8')
start=s.index('        if self.selector is not None or self.lilicorr is not None:\n',s.index('        draft_plan.bind(forward_batch, draft_translator)'))
end=s.index('        draft_tokens = self._draft_block_tokens_buf[:bs]',start)
old=s[start:end]
assert 'draft_model_runner.forward(forward_batch)' in old
new="""        # Isolated research hook: no installer or active change by default.
        lookup_proposer = getattr(self, '_request_lookup_proposer', None)
        lookup_plan = (lookup_proposer(worker=self, batch=batch,
                                      anchor_tokens=block_ids[:, 0],
                                      prefix_lens=prefix_lens)
                       if lookup_proposer is not None and get_parallel().tp_size == 1\n                       and get_parallel().pp_size == 1 and not get_parallel().attn_dp_enabled else None)
        if lookup_plan is not None and lookup_plan.all_hit:
            self._selector_sample = None
            draft_next = lookup_plan.rows
        else:
"""+''.join('    '+line if line.strip() else line for line in old.splitlines(True))+"""            if lookup_plan is not None:
                draft_next = lookup_plan.merge(draft_next)

"""
out=s[:start]+new+s[end:]
ast.parse(out)
(P/'dflash_worker_lookup_overlay.py').write_text(out,encoding='utf-8')
diff=''.join(difflib.unified_diff(s.splitlines(True),out.splitlines(True),fromfile='a/python/sglang/srt/speculative/dflash_worker_v2.py',tofile='b/python/sglang/srt/speculative/dflash_worker_v2.py'))
(P/'lookup_default_off.diff').write_text(diff,encoding='utf-8')
(P/'OVERLAY_MANIFEST.json').write_text(json.dumps({'status':'DEFAULT_OFF_ISOLATED_OVERLAY_NOT_INSTALLED','base_sha256':hashlib.sha256(s.encode()).hexdigest(),'overlay_sha256':hashlib.sha256(out.encode()).hexdigest(),'preserved':['embedding and compact draft cache rebuild','draft translator plan binding','all target verify and target hidden KV append','fixed8 slot reservation and output stride'],'held':['Native TP8 rank consensus and installation','Real target/draft KV equivalence and subsequent continuation','Host snapshot provider and incremental history wiring','Full step cost including CPU readbacks and lookup overhead']},indent=2)+'\n')
print(json.dumps({'status':'BUILT_DEFAULT_OFF_LOOKUP_OVERLAY','diff_bytes':len(diff)}))
