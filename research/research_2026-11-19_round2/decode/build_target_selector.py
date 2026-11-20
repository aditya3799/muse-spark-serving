"""Separate default-OFF native target post-gather selector integration."""
import ast,difflib,hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1]
helper=here.parent/'coordination/target_argmax_postops/target_select.py'
expected='bbb25013e68da53b5449646c81ac01367d0647919febde248e60cff50f30023f'
assert hashlib.sha256(helper.read_bytes()).hexdigest()==expected,'root helper changed; review/rebind before building'
base=(here/'dflash_sources/logits_processor.py').read_text(encoding='utf-8')
old='''        best_rank = gathered[..., 0].argmax(dim=0, keepdim=True)
        return gathered[..., 1].gather(0, best_rank).squeeze(0).long()'''
new='''        if os.environ.get("SGLANG_EXPERIMENTAL_DFLASH_TARGET_SELECT", "0") == "1":
            from sglang.kernels.ops.sampling.dflash_target_select import fused_select
            return fused_select(gathered)
'''+old
assert base.count(old)==1
source=base.replace('import logging\n','import logging\nimport os\n',1).replace(old,new)
ast.parse(source)
files={'python/sglang/srt/layers/logits_processor.py':(base,source),
 'python/sglang/kernels/ops/sampling/dflash_target_select.py':('',helper.read_text(encoding='utf-8'))}
out=here/'target_selector_integration';out.mkdir(exist_ok=True);diff=[]
for name,(before,after) in files.items():
 path=out/'sources'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(after.encode())
 diff.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+name if before else '/dev/null',tofile='b/'+name))
delta=''.join(diff);parent=(root/'release_2026-11-18/candidate.diff').read_text(encoding='utf-8')
candidate=parent+('' if parent.endswith('\n') else '\n')+delta
(out/'delta.diff').write_bytes(delta.encode());(out/'candidate.diff').write_bytes(candidate.encode())
(out/'manifest.json').write_text(json.dumps(dict(applied=False,default_enabled=False,helper_sha256=expected,
 candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),files={name:hashlib.sha256(after.encode()).hexdigest() for name,(_,after) in files.items()},
 scope='Only native deferred_argmax final argmax/gather/long; native softcap, float32 payload and one collective unchanged.',
 local_eager_performance='inconclusive controls; no KEEP',serving_performance='unmeasured'),indent=2))
print('prepared isolated target selector',len(candidate),'bytes')
