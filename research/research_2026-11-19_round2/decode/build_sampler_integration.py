"""Unapplied opt-in patch around the frozen ordinary DFlash greedy sampler."""
import ast,difflib,hashlib,json,subprocess
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1]
base=(here/'dflash_sources/dflash_worker_v2.py').read_text(encoding='utf-8');source=base
before='        self.out = torch.empty((max_tokens,), dtype=torch.int64, device=device)'
after=before+'''
        from sglang.kernels.ops.speculative.packed_draft_sampler import maybe_prepare_packed
        self._packed_argmax = maybe_prepare_packed(
            tp_group=tp_group, max_tokens=max_tokens, dtype=weight.dtype, device=device,
            native_id_range=(self.org_vocab_start, self.org_vocab_start + self.num_org),
        )'''
assert source.count(before)==1;source=source.replace(before,after)
before='''        gathered_max = self.gathered_max[: self.tp_size * n]
        gathered_ids = self.gathered_ids[: self.tp_size * n]'''
after='''        if self._packed_argmax is not None:
            self._packed_argmax.run(local_max, local_arg, self.out[:n], self.tp_group)
            return
'''+before
assert source.count(before)==1;source=source.replace(before,after);ast.parse(source)
files={'python/sglang/srt/speculative/dflash_worker_v2.py':(base,source),
 'python/sglang/kernels/ops/speculative/packed_draft_sampler.py':('',(here/'packed_sampler.py').read_text().replace('from packed_sampler_kernel import','from .packed_draft_sampler_kernel import')),
 'python/sglang/kernels/ops/speculative/packed_draft_sampler_kernel.py':('',(here/'packed_sampler_kernel.py').read_text())}
out=here/'sampler_integration';out.mkdir(exist_ok=True);diff=[]
for name,(old,new) in files.items():
 p=out/'sources'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(new.encode('utf-8'))
 diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+name if old else '/dev/null',tofile='b/'+name))
delta=''.join(diff);(out/'delta.diff').write_bytes(delta.encode('utf-8'))
parent=(root/'release_2026-11-18/candidate.diff').read_text(encoding='utf-8');candidate=parent+('\n' if not parent.endswith('\n') else '')+delta
(out/'candidate.diff').write_bytes(candidate.encode('utf-8'))
(out/'manifest.json').write_text(json.dumps(dict(applied=False,default_enabled=False,candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
 files={k:hashlib.sha256(v[1].encode()).hexdigest() for k,v in files.items()},scope='Ordinary dense greedy DFlash draft sampler only; unchanged logits/max; onepackedallgather replacingtwo'),indent=2))
print('unapplied sampler candidate',len(candidate),'bytes')
