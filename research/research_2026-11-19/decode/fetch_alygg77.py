"""Bounded attention-only primary source snapshot; never executes downloaded code."""
import hashlib,json,urllib.request
from pathlib import Path
commit='6cae49a639ee6d14b929a5fd4a63c627f96d7dc0'
out=Path(__file__).resolve().parent/'sources/alygg77';out.mkdir(parents=True,exist_ok=True)
records=[]
for name in ['kernels/muse_attention_decode.py','RESULTS.md','NOTICE','docs/H200_MEASURED_FACTS.md','docs/NEGATIVE_RESULTS.md']:
 url=f'https://raw.githubusercontent.com/alygg77/okc-muse-spark-kernels/{commit}/{name}'
 with urllib.request.urlopen(url,timeout=30) as response:data=response.read(2_000_001)
 assert len(data)<=2_000_000
 target=out/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
 records.append(dict(url=url,commit=commit,path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
(out/'manifest.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
