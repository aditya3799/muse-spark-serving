"""Bounded primary source snapshot, pinned to root's independently resolved HEAD."""
import hashlib,json,urllib.request
from pathlib import Path
here=Path(__file__).resolve().parent;out=here/'latest_dflash';out.mkdir(exist_ok=True)
pin='a735253e8433d5fc57bf585b3eef93e4b4cfa885'
paths=['python/sglang/srt/speculative/dflash_worker_v2.py','python/sglang/srt/speculative/dflash_utils.py','python/sglang/srt/layers/logits_processor.py','python/sglang/srt/speculative/spec_tp_sync.py']
records=[]
for source in paths:
 url='https://raw.githubusercontent.com/sgl-project/sglang/'+pin+'/'+source
 try:
  with urllib.request.urlopen(url,timeout=25) as response:
   blob=response.read(1500000)
   assert len(blob)<1500000,'bounded source response exceeded1.5MB'
  path=out/Path(source).name;path.write_bytes(blob)
  records.append(dict(source=source,url=url,sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob)))
 except Exception as exc:records.append(dict(source=source,url=url,error=str(exc)))
(out/'manifest.json').write_text(json.dumps(dict(pin=pin,files=records),indent=2));print(json.dumps(records,indent=2))
