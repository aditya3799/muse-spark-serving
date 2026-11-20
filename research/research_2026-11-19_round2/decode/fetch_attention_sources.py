"""Bounded primary-source snapshot, using commits already pinned by root audit."""
import concurrent.futures,datetime,hashlib,json,urllib.request
from pathlib import Path
here=Path(__file__).resolve().parent;destination=here/'attention_sources';destination.mkdir(exist_ok=True)
targets=[('Dao-AILab/flash-attention','68bd095cdd338c5171bccb9c200529f3620b85c9',p) for p in ['flash_attn/cute/interface.py','flash_attn/cute/flash_fwd_sm90.py','flash_attn/cute/flash_fwd.py']]
targets += [('flashinfer-ai/flashinfer','855eece4e9b5b796a46148664a3e27da571cd98b',p) for p in ['flashinfer/decode.py','flashinfer/prefill.py']]
def fetch(target):
 repo,pin,path=target;url=f'https://raw.githubusercontent.com/{repo}/{pin}/{path}'
 record=dict(repo=repo,pin=pin,path=path,url=url)
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'OpenKernelChallenge-research'}),timeout=30) as response:
   data=response.read(1024*1024+1);assert len(data)<=1024*1024
  name=repo.replace('/','__')+'__'+path.replace('/','__');(destination/name).write_bytes(data)
  record.update(local=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 except Exception as e:record['error']=str(e)
 return record
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,targets))
(destination/'manifest.json').write_text(json.dumps(dict(as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=records),indent=2))
print(json.dumps(records,indent=2))
