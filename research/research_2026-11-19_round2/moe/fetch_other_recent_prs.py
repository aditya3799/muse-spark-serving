import pathlib,urllib.request,json,hashlib,concurrent.futures,datetime
p=pathlib.Path(__file__).resolve().parent/'sources/nov10_sweep/other_engines';p.mkdir(exist_ok=True)
jobs=[('InternLM/lmdeploy',5142),('InternLM/lmdeploy',5161),('InternLM/lmdeploy',5174),('huggingface/kernels-community',1240),('flashinfer-ai/flashinfer',9581)]
def fetch(job):
 repo,n=job;root='https://api.github.com/repos/'+repo+'/pulls/'+str(n);row={'repo':repo,'number':n}
 for leaf,url in [('metadata',root),('files',root+'/files?per_page=100')]:
  req=urllib.request.Request(url,headers={'User-Agent':'okc-read-only-research'});b=urllib.request.urlopen(req,timeout=30).read();f=p/(repo.replace('/','_')+'_'+str(n)+'_'+leaf+'.json');f.write_bytes(b);d=json.loads(b);row[leaf+'_sha256']=hashlib.sha256(b).hexdigest()
  if leaf=='metadata':row.update(title=d['title'],head=d['head']['sha'],created=d['created_at'],updated=d['updated_at'],merged=d['merged'],body=d.get('body'),url=d['html_url'])
  else:row['files']=[dict(path=x['filename'],status=x['status'],additions=x['additions']) for x in d]
 return row
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:r=list(pool.map(fetch,jobs))
(p/'manifest.json').write_text(json.dumps(dict(retrieved_utc=datetime.datetime.utcnow().isoformat()+'Z',items=r),indent=2));print(json.dumps(r,indent=2))
