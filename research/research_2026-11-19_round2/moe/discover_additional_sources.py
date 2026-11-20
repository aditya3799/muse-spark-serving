import concurrent.futures,json,urllib.request,hashlib
from pathlib import Path
here=Path(__file__).resolve().parent/'sources';here.mkdir(exist_ok=True)
repos=[('13thblade/ninfer-all-nvfp4','9b4b8a00760e001191d0177b7f2651b882e77de8'),('Avarok-Cybersecurity/atlas','eae65c07a0daa8073a9edff77199a45254bfa34a')]
def one(item):
 repo,commit=item;url=f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1'
 request=urllib.request.Request(url,headers={'User-Agent':'OpenKernelChallenge-research'})
 with urllib.request.urlopen(request,timeout=30) as r:data=r.read()
 file=repo.replace('/','__')+'__tree.json';(here/file).write_bytes(data);tree=json.loads(data)
 files=[x['path'] for x in tree['tree'] if x.get('type')=='blob' and any(s in x['path'].lower() for s in ('nvfp4','w4a16','moe','hopper')) and x['path'].split('.')[-1] in ('cu','cuh','h','hpp','cpp','py','md','rs')]
 return dict(repository=repo,commit=commit,tree_file=file,truncated=tree.get('truncated'),matching_files=files,sha256=hashlib.sha256(data).hexdigest())
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(one,repos))
(here/'additional_round2_sources.json').write_text(json.dumps(rows,indent=2))
for row in rows:print(json.dumps(row))
