"""Bounded primary PR/issue sweep; download metadata only, execute no external code."""
import urllib.request,urllib.parse,json,hashlib,concurrent.futures,datetime
from pathlib import Path
here=Path(__file__).resolve().parent;out=here/'sources/nov10_sweep';out.mkdir(exist_ok=True)
repos=['flashinfer-ai/flashinfer','vllm-project/vllm','NVIDIA/TensorRT-LLM','sgl-project/sglang']
queries=[('nvfp4_hopper_'+r.split('/')[-1],f'repo:{r} nvfp4 hopper updated:>=2026-11-10') for r in repos]
queries += [('w4a16_recent','w4a16 moe updated:>=2026-11-10'),('nvfp4_g16_recent','nvfp4 group_size 16 updated:>=2026-11-10')]
def fetch(job):
 name,q=job;url='https://api.github.com/search/issues?'+urllib.parse.urlencode(dict(q=q,sort='updated',order='desc',per_page=50))
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'okc-source-audit','Accept':'application/vnd.github+json'})
  raw=urllib.request.urlopen(req,timeout=30).read();d=json.loads(raw);p=out/(name+'.json');p.write_bytes(raw)
  return dict(name=name,query=q,url=url,file=p.name,sha256=hashlib.sha256(raw).hexdigest(),total=d.get('total_count'),incomplete=d.get('incomplete_results'),items=[dict(title=x['title'],url=x['html_url'],number=x['number'],repository=x['repository_url'],updated=x['updated_at'],created=x['created_at'],state=x['state'],pr='pull_request' in x) for x in d.get('items',[])])
 except Exception as e:return dict(name=name,query=q,error=repr(e))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(fetch,queries))
r=dict(retrieved_utc=datetime.datetime.utcnow().isoformat()+'Z',since='2026-11-10',bounded=True,queries=rows)
(out/'manifest.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
