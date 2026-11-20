import concurrent.futures, datetime, hashlib, json, pathlib, subprocess, urllib.request
out=pathlib.Path(__file__).resolve().parent/'focused_sources';out.mkdir(exist_ok=True)
repos={'cloudflareresearch/unweight-kernels':['README.md'], 'dphnAI/nf12':['README.md','docs/SPEC.md'], 'parda21/NWC':['README.md','docs/results.md'], 'WeightLab/bf16-exponent-compression':['README.md','METHODOLOGY.md'], 'cenconq25/Turbo-Lossless':['README.md'], 'brianbell-x/weight-compression':['README.md'], 'lightseekorg/tokenspeed':['python/tokenspeed/runtime/models/muse_spark.py','python/tokenspeed/runtime/models/dflash2.py','python/tokenspeed/runtime/execution/drafter/dflash2.py','docs/recipes/models.md','tokenspeed-scheduler/csrc/cache/coordinator/admission_planner.h','tokenspeed-scheduler/csrc/scheduler/capacity_model.h','tokenspeed-scheduler/csrc/scheduler/operations/prefill_chunk.h']}
def one(item):
 repo,paths=item;r={'repo':repo,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':[]}
 try:
  sha=subprocess.check_output(['git','ls-remote','https://github.com/'+repo+'.git','HEAD'],text=True,timeout=35).split()[0];r['commit']=sha
  for path in paths:
   f={'path':path,'url':f'https://raw.githubusercontent.com/{repo}/{sha}/{path}'}
   try:
    data=urllib.request.urlopen(f['url'],timeout=25).read(400001)
    if len(data)>400000:raise ValueError('file budget exceeded')
    target=out/(repo.replace('/','__')+'__'+path.replace('/','__'));target.write_bytes(data)
    f.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),local=str(target.name),status='retrieved')
   except Exception as e:f.update(status='unavailable',error=str(e)[:250])
   r['files'].append(f)
 except Exception as e:r['error']=str(e)[:250]
 print(repo,[(f['path'],f['status']) for f in r['files']],flush=True);return r
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(one,repos.items()))
(out.parent/'focused_manifest.json').write_text(json.dumps(records,indent=2))
