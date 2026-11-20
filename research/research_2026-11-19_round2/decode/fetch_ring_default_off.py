import hashlib,json,urllib.request
from pathlib import Path
out=Path(__file__).resolve().parent/'ring_default_off';out.mkdir(exist_ok=True)
pin='981becd7afebb330891492bdff0ca8cd59f8b616'
url='https://github.com/sgl-project/sglang/commit/'+pin+'.patch'
with urllib.request.urlopen(url,timeout=30) as r:data=r.read(200001)
assert len(data)<=200000
(out/'merged.patch').write_bytes(data)
(out/'manifest.json').write_text(json.dumps(dict(commit=pin,url=url,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),
 status='merged Nov19; not closed-unmerged',pr='https://github.com/sgl-project/sglang/pull/51386'),indent=2))
print(len(data),'bytes pinned')
