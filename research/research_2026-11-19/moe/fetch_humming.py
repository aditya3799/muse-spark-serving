import json,urllib.request,hashlib,concurrent.futures,time
from pathlib import Path
base=Path(__file__).resolve().parent
root=base/'upstream_humming';root.mkdir(exist_ok=True)
j=json.loads((base/'humming_tree.json').read_text());sha=j['sha']
files=[x for x in j['tree'] if x['type']=='blob' and (x['path'].startswith('humming/') or x['path'] in ('tests/test_sm90_heuristics.py','docs/config.md'))]
assert len(files)<200 and sum(x.get('size',0) for x in files)<3000000
def fetch(x):
 p=x['path'];target=root/p;target.parent.mkdir(exist_ok=True,parents=True)
 if target.exists():return {'path':p,'bytes':target.stat().st_size,'cached':True}
 url='https://raw.githubusercontent.com/vllm-project/humming/'+sha+'/'+p
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'okc-research'}),timeout=20) as f:data=f.read(300000)
  assert len(data)<300000
  target.write_bytes(data);return {'path':p,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 except Exception as e:return {'path':p,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(fetch,files))
(base/'humming_source_manifest.json').write_text(json.dumps({'revision':sha,'files':results},indent=2))
print(json.dumps({'files':len(results),'bytes':sum(x.get('bytes',0) for x in results),'errors':[x for x in results if 'error'in x]}))
