"""Separate unapplied target-greedy broadcast delta, never merged with sampler."""
import ast,difflib,hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1]
base=(here/'dflash_sources/dflash_worker_v2.py').read_text(encoding='utf-8');source=base
edits=[
('        self._target_logits_processor = self._defer_target_verify_gather()',
 '''        self._target_logits_processor = self._defer_target_verify_gather()
        from sglang.srt.speculative.deferred_argmax_agreement import prepare_deferred_agreement
        self._deferred_agreement = prepare_deferred_agreement(
            self._target_logits_processor, self._tp_sync
        )'''),
('        target_predict: Optional[torch.Tensor] = None,\n    ):',
 '        target_predict: Optional[torch.Tensor] = None,\n        target_agreement=None,\n    ):'),
('''            target_predict = target_predict.view(bs, int(self.block_size))
            self._tp_sync.sync(SpecTpSyncSite.DFLASH_ACCEPT_GREEDY, target_predict)''',
 '''            agreed = False
            if target_agreement is not None:
                agreed = self._deferred_agreement.consume(target_agreement, target_predict, self._tp_sync)
            target_predict = target_predict.view(bs, int(self.block_size))
            if not agreed:
                self._tp_sync.sync(SpecTpSyncSite.DFLASH_ACCEPT_GREEDY, target_predict)'''),
('        target_predict = None\n        target_lp = self._target_logits_processor',
 '        target_predict = None\n        target_agreement = None\n        target_lp = self._target_logits_processor'),
('''                target_predict = target_lp.deferred_argmax(
                    logits_output.next_token_logits
                )''',
 '''                if self._deferred_agreement is None:
                    target_predict = target_lp.deferred_argmax(
                        logits_output.next_token_logits
                    )
                else:
                    target_predict, target_agreement = self._deferred_agreement.produce(
                        target_lp, self._tp_sync, logits_output.next_token_logits
                    )'''),
('''            target_predict=target_predict,
        )

        sampling_mask_capture''',
 '''            target_predict=target_predict,
            target_agreement=target_agreement,
        )

        sampling_mask_capture''')]
for before,after in edits:
 assert source.count(before)==1,(before,source.count(before));source=source.replace(before,after)
ast.parse(source)
files={'python/sglang/srt/speculative/dflash_worker_v2.py':(base,source),
 'python/sglang/srt/speculative/deferred_argmax_agreement.py':('',(here/'deferred_sync.py').read_text(encoding='utf-8'))}
out=here/'deferred_sync_integration';out.mkdir(exist_ok=True);diff=[]
for name,(old,new) in files.items():
 p=out/'sources'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(new.encode())
 diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+name if old else '/dev/null',tofile='b/'+name))
delta=''.join(diff);(out/'delta.diff').write_bytes(delta.encode())
parent=(root/'release_2026-11-18/candidate.diff').read_text(encoding='utf-8');candidate=parent+('' if parent.endswith('\n') else '\n')+delta
(out/'candidate.diff').write_bytes(candidate.encode())
(out/'manifest.json').write_text(json.dumps(dict(applied=False,default_enabled=False,performance_measured=False,candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),files={k:hashlib.sha256(v[1].encode()).hexdigest() for k,v in files.items()}),indent=2))
print('unapplied deferred sync candidate',len(candidate))
