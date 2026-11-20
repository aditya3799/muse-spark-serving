import urllib.request,json,hashlib,datetime
from pathlib import Path
here=Path(__file__).resolve().parent
req=urllib.request.Request('https://api.github.com/repos/sgl-project/sglang/commits/main',headers={'User-Agent':'okc-source-audit'})
head=json.loads(urllib.request.urlopen(req,timeout=30).read())['sha']
path='docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx'
url='https://raw.githubusercontent.com/sgl-project/sglang/'+head+'/'+path
raw=urllib.request.urlopen(url,timeout=30).read();p=here/'sources/sglang_deepseek_v4_cookbook.mdx';p.write_bytes(raw)
r=dict(repository='sgl-project/sglang',commit=head,path=path,url=url,file=p.name,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),retrieved_utc=datetime.datetime.utcnow().isoformat()+'Z',scope='Documentation check only, not a runtime proof',finding='SM90 CUTLASS W4A16 recipe concerns original MXFP4 checkpoints; documented NVFP4 hybrid path requires SM100+. New low-latency Humming option quantizes activations toFP8, outside original W4A16 constraint. No new exactNVFP4 group16 backend established by this cookbook.')
(here/'sources/sglang_cookbook_manifest.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
