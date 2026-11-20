import json,pathlib,urllib.request,concurrent.futures,hashlib
p=pathlib.Path('serving/research_2026-11-19_round2/moe/sources/nov10_sweep/other_engines');jobs=[]
for path in ['lmdeploy/turbomind/weight_format.py','src/turbomind/kernels/gemm/arch/config_sm80_s16816.h','src/turbomind/kernels/gemm/kernel/nvfp4.h']:
 jobs.append(('lmdeploy5173','InternLM/lmdeploy','6f0dc62301d2e13a59499a95659877a4122450b4',path))
for path in ['include/flashinfer/fused_moe/fused_moe_w4a16_mxfp4_latent_sm90/kernels.cuh','flashinfer/fused_moe/w4a16_mxfp4_moe.py']:
 jobs.append(('flashinfer9581','flashinfer-ai/flashinfer','fd6d8549c6bdac3d3a748663711fa1c1c1f9413c',path))
for path in ['formats.py','mma.py','grouped.py','compat.py']:
 jobs.append(('finegrained1240','huggingface/kernels-community','cbd5b592734823dd0f9eab1fd006c41e1a912033','finegrained-kernels/torch-ext/finegrained_kernels/'+path))
def get(j):
 name,repo,head,path=j;u='https://raw.githubusercontent.com/'+repo+'/'+head+'/'+path;b=urllib.request.urlopen(u,timeout=30).read();assert len(b)<1000000;d=p/name;d.mkdir(exist_ok=True);f=d/pathlib.Path(path).name;f.write_bytes(b);return dict(file=str(f),url=u,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:r=list(pool.map(get,jobs))
(p/'leaf_manifest.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
