import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
here=Path(__file__).resolve().parent/'sources';here.mkdir(exist_ok=True)
commit='855eece4e9b5b796a46148664a3e27da571cd98b'
paths=['flashinfer/fused_moe/api.py','flashinfer/fused_moe/runners.py','flashinfer/fused_moe/prepare.py','docs/design_docs/frost_low_latency_moe.md']
def fetch(path):
 url='https://raw.githubusercontent.com/flashinfer-ai/flashinfer/'+commit+'/'+path
 try:
  with urllib.request.urlopen(url,timeout=25) as r: data=r.read()
  name='flashinfer__'+path.replace('/','__');(here/name).write_bytes(data)
  return dict(path=path,url=url,file=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 except Exception as exc:return dict(path=path,error=str(exc))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: rows=list(pool.map(fetch,paths))
(here/'frost_manifest.json').write_text(json.dumps(dict(commit=commit,files=rows),indent=2))
print(json.dumps(rows))
