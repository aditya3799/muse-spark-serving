import hashlib,json,urllib.request
from pathlib import Path
here=Path(__file__).resolve().parent/'sources';commit='855eece4e9b5b796a46148664a3e27da571cd98b'
path='flashinfer/fused_moe/backends/frost_low_latency/fma/fp4_weight.py'
url='https://raw.githubusercontent.com/flashinfer-ai/flashinfer/'+commit+'/'+path
with urllib.request.urlopen(url,timeout=20) as r:data=r.read()
file='flashinfer__'+path.replace('/','__');(here/file).write_bytes(data)
row=dict(path=path,url=url,file=file,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
p=here/'frost_manifest.json';d=json.loads(p.read_text());d['files'].append(row);p.write_text(json.dumps(d,indent=2));print(json.dumps(row))
