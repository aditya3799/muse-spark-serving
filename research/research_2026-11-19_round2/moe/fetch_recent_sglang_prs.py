import urllib.request,json,hashlib,concurrent.futures,datetime
from pathlib import Path
here=Path(__file__).resolve().parent;out=here/'sources/nov10_sweep';ids=[50800,50801,50802,51227,51393,51390,51302]
def fetch(n):
 result={'number':n}
 for suffix,label in [('', 'metadata'),('/files?per_page=100','files')]:
  url=f'https://api.github.com/repos/sgl-project/sglang/pulls/{n}'+suffix
  try:
   raw=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'okc-source-audit'}),timeout=30).read();d=json.loads(raw)
   p=out/f'sglang_pr{n}_{label}.json';p.write_bytes(raw);result[label+'_sha256']=hashlib.sha256(raw).hexdigest()
   if label=='metadata':result.update(title=d['title'],state=d['state'],merged=d['merged'],merged_at=d['merged_at'],created_at=d['created_at'],updated_at=d['updated_at'],head=d['head']['sha'],base=d['base']['sha'],merge_commit=d['merge_commit_sha'],body=d['body'],url=d['html_url'])
   else:result['files']=[dict(filename=x['filename'],status=x['status'],additions=x['additions'],deletions=x['deletions'],raw_url=x.get('raw_url')) for x in d]
  except Exception as e:result[label+'_error']=repr(e)
 return result
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(fetch,ids))
r=dict(retrieved_utc=datetime.datetime.utcnow().isoformat()+'Z',prs=rows);(out/'sglang_pr_manifest.json').write_text(json.dumps(r,indent=2))
for x in rows:print(json.dumps({k:v for k,v in x.items() if k!='body'}))
