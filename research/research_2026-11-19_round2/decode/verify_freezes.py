"""Read-only rehash of frozen decode packages after independent research work."""
import hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent;rows=[]
for name in ('prologue_freeze_manifest.json','sampler_freeze_manifest.json','deferred_sync_freeze_manifest.json','target_selector_freeze_manifest.json'):
 path=here/name;manifest=json.loads(path.read_text());checked=[]
 for relative,expected in manifest.get('files',{}).items():
  target=here/relative
  if not isinstance(expected,str):continue
  actual=hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
  checked.append(dict(path=relative,exact=actual==expected,actual=actual,expected=expected))
 for relative,expected in manifest.get('gpu_proofs',{}).items():
  target=here.parent/relative;actual=hashlib.sha256(target.read_bytes()).hexdigest()
  checked.append(dict(path=relative,exact=actual==expected,actual=actual,expected=expected))
 rows.append(dict(manifest=name,manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),files=len(checked),all_exact=all(x['exact'] for x in checked),mismatches=[x for x in checked if not x['exact']]))
out=dict(all_exact=all(x['all_exact'] for x in rows),packages=rows,original_files_mutated=False)
(here/'freeze_recheck.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
assert out['all_exact']
