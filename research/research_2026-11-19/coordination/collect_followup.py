import concurrent.futures,datetime,hashlib,json,pathlib,subprocess,urllib.request
root=pathlib.Path(__file__).resolve().parent; out=root/'followup_sources';out.mkdir(exist_ok=True)
repos={'PaddlePaddle/FastDeploy':['README_CN.md'],'cheetah-lang/flashformer':['README.md'],'NVIDIA/hyperflare-inference':['docs/core-design.md','ignition/recipes/muse_spark/recipe.py']}
def one(item):
 repo,paths=item;r={'repo':repo,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':[]}
 try:
  sha=subprocess.check_output(['git','ls-remote','https://github.com/'+repo+'.git','HEAD'],text=True,timeout=30).split()[0];r['commit']=sha
  for path in paths:
   f={'path':path,'url':f'https://raw.githubusercontent.com/{repo}/{sha}/{path}'}
   try:
    data=urllib.request.urlopen(f['url'],timeout=20).read(400001)
    if len(data)>400000:raise ValueError('file budget exceeded')
    target=out/(repo.replace('/','__')+'__'+path.replace('/','__'));target.write_bytes(data)
    f.update(status='retrieved',bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),local=target.name)
   except Exception as e:f.update(status='unavailable',error=str(e)[:200])
   r['files'].append(f)
 except Exception as e:r['error']=str(e)[:200]
 print(repo,[(f['path'],f['status']) for f in r['files']],flush=True);return r
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(one,repos.items()))
(root/'followup_manifest.json').write_text(json.dumps(results,indent=2))
