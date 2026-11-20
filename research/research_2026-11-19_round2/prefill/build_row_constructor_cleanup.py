"""Isolated constructor ownership amendment after reviewed failure policy."""
import ast,difflib,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[1]
P=H.parent/'coordination/projection_failure_policy'
rel=Path('python/sglang/srt/speculative/dflash_projection_row.py')
before=(P/'sources'/rel).read_text()
factory='def create_prefill_projection(model,group,capture_rows,*,ordinary_tp,measure=True):'
assert before.count(factory)==1
before=before.replace(factory,'@torch.no_grad()\n'+factory)
old='    plan=RowPlan(lib,handle,world,x,w,bytes_)\n'
new='''    try:
        plan=RowPlan(lib,handle,world,x,w,bytes_)
    except Exception:
        # Ownership transfers only after the Python constructor completes.
        lib.row_plan_destroy(handle)
        raise
'''
assert before.count(old)==1
changed=before.replace(old,new)
old='''        # Even an unexpected local discovery failure joins this CPU collective.
        # Peers therefore fall back together instead of entering a GPU gather.
'''
new='''        # Fatal discovery errors were voted above and abort every rank.
        # Recoverable allocation/plan rejection reaches coherent acceptance.
'''
assert changed.count(old)==1
changed=changed.replace(old,new);ast.parse(changed)
out=H/'row_constructor_cleanup_unapplied';dest=out/'sources'/rel
dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(changed)
diff=''.join(difflib.unified_diff(before.splitlines(True),changed.splitlines(True),fromfile='a/'+rel.as_posix(),tofile='b/'+rel.as_posix()))
(out/'incremental.diff').write_text(diff)
manifest=dict(status='UNAPPLIED cleanup-only amendment; no GPU or numeric change',relative_to='joint530e +failurepolicy71f6152c +early74d63805 +no_grad07f4c518',
 patch_sha256=hashlib.sha256(diff.encode()).hexdigest(),bytes=len(diff.encode()),
 source_before_sha256=hashlib.sha256(before.encode()).hexdigest(),source_after_sha256=hashlib.sha256(changed.encode()).hexdigest())
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
