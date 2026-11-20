"""Pin two released Hopper donors; bounded source only, no installation."""
import hashlib,json,urllib.request
from pathlib import Path
out=Path(__file__).resolve().parent/'flashinfer_nov';out.mkdir(exist_ok=True)
items=[('5188','0bb4278501085a09054df3156400ea25ed28fec9','flashinfer/prefill.py','prefill.py'),
 ('5617','2cb3521e52d0452a2524bad7d2c4d43a273a2aba','include/flashinfer/sampling.cuh','sampling.cuh')]
rows=[]
for pr,pin,path,name in items:
 url='https://raw.githubusercontent.com/flashinfer-ai/flashinfer/'+pin+'/'+path
 with urllib.request.urlopen(url,timeout=40) as r:data=r.read(900001)
 assert len(data)<=900000
 (out/name).write_bytes(data)
 rows.append(dict(pr=pr,pin=pin,url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),path=name))
(out/'manifest.json').write_text(json.dumps(dict(release='v0.7.4 Nov10/11',primary_release='https://flashinfer.ai/releases/',files=rows),indent=2))
print([(x['pr'],x['bytes']) for x in rows])
