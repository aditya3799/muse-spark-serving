import hashlib,json,urllib.request
from pathlib import Path
here=Path(__file__).resolve().parent/'sources';commit='855eece4e9b5b796a46148664a3e27da571cd98b'
paths=['flashinfer/fused_moe/backends/frost_low_latency/fp4_weight_only.py','flashinfer/fused_moe/backends/frost_low_latency/runner.py','flashinfer/fused_moe/backends/frost_low_latency/fma/activation_config.py']
rows=[]
for path in paths:
 url='https://raw.githubusercontent.com/flashinfer-ai/flashinfer/'+commit+'/'+path
 try:
  with urllib.request.urlopen(url,timeout=20) as r:data=r.read()
  file='flashinfer__'+path.replace('/','__');(here/file).write_bytes(data)
  rows.append(dict(path=path,url=url,file=file,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
 except Exception as exc:rows.append(dict(path=path,error=str(exc)))
p=here/'frost_manifest.json';d=json.loads(p.read_text());d['files']+=rows;p.write_text(json.dumps(d,indent=2));print(json.dumps(rows))
