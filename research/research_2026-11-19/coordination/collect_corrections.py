import concurrent.futures, datetime, hashlib, json, pathlib, subprocess, urllib.request
root=pathlib.Path(__file__).resolve().parent; out=root/'corrected_sources';out.mkdir(exist_ok=True)
repos={'ai-dynamo/dynamo':['README.md'],'microsoft/sarathi-serve':['README.md'],'thu-pacman/chitu':['docs/en/README.md','docs/en/SUPPORTED_MODELS.md'],'vectorch-ai/ScaleLLM':['README.md'],'lightseekorg/tokenspeed':['python/tokenspeed/runtime/models/dflash.py','tokenspeed-scheduler/csrc/scheduler/capacity_model.cpp']}
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
(root/'corrected_manifest.json').write_text(json.dumps(results,indent=2))
state=json.loads((root/'SESSION_STATE.json').read_text(encoding='utf-8-sig'))
state['remote']['new_jobs_this_session']=1
state['remote']['cache_check']='f464d5f09f534363'
state['remote']['remaining_after_launch']='1 scored and2 check expected; verify with CLI'
(root/'SESSION_STATE.json').write_text(json.dumps(state,indent=2))
