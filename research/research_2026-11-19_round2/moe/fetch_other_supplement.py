import urllib.request,pathlib,hashlib,json
p=pathlib.Path('serving/research_2026-11-19_round2/moe/sources/nov10_sweep/other_engines');jobs=[('lmdeploy5173','InternLM/lmdeploy','6f0dc62301d2e13a59499a95659877a4122450b4','src/turbomind/kernels/gemm/kernel/sm80_16816_4.cu'),('finegrained1240','huggingface/kernels-community','cbd5b592734823dd0f9eab1fd006c41e1a912033','finegrained-kernels/torch-ext/finegrained_kernels/epilogue.py')];r=[]
for name,repo,head,path in jobs:
 u='https://raw.githubusercontent.com/'+repo+'/'+head+'/'+path;b=urllib.request.urlopen(u,timeout=30).read();f=p/name/pathlib.Path(path).name;f.write_bytes(b);r.append(dict(file=str(f),url=u,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
(p/'supplement_leaf_manifest.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
