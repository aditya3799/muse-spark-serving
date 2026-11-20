import urllib.request, pathlib, hashlib, json, time
root=pathlib.Path(r'D:\OpenKernelChallenge\serving\research_competitive_2026-11-13\sources')
urls={
'vllm_64077.diff':'https://github.com/vllm-project/vllm/pull/64077.diff',
'vllm_65438.diff':'https://github.com/vllm-project/vllm/pull/65438.diff',
'vllm_64553.diff':'https://github.com/vllm-project/vllm/pull/64553.diff',
'okc_home.html':'https://openkernelchallenge.dev/',
'sglang_43363.diff':'https://github.com/sgl-project/sglang/pull/43363.diff',
}
from concurrent.futures import ThreadPoolExecutor

def fetch(item):
 name,url=item
 try:
  request=urllib.request.Request(url, headers={'User-Agent':'serving-research/1.0'})
  data=urllib.request.urlopen(request,timeout=35).read()
  (root/name).write_bytes(data)
  return dict(name=name,url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 except Exception as e:return dict(name=name,url=url,error=str(e))
with ThreadPoolExecutor(max_workers=5) as pool: results=list(pool.map(fetch,urls.items()))
(root/'manifest.json').write_text(json.dumps(results,indent=2))
for x in results:print(json.dumps(x))
