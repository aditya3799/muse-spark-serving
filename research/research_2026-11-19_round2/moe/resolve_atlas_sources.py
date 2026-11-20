import concurrent.futures,json,urllib.request,hashlib,posixpath
from pathlib import Path
here=Path(__file__).resolve().parent/'sources';p=here/'additional_leaf_manifest.json';rows=json.loads(p.read_text());items=[]
for row in rows:
 if row.get('bytes',1000)<100 and 'file' in row:
  target=(here/row['file']).read_text().strip();resolved=posixpath.normpath(posixpath.join(posixpath.dirname(row['path']),target))
  items.append((row['repository'],row['commit'],resolved,row['path']))
def one(item):
 repo,commit,path,link=item;url=f'https://raw.githubusercontent.com/{repo}/{commit}/{path}'
 with urllib.request.urlopen(url,timeout=20) as r:data=r.read()
 file=repo.split('/')[0]+'__'+path.replace('/','__');(here/file).write_bytes(data)
 return dict(repository=repo,commit=commit,path=path,symlink_from=link,file=file,bytes=len(data),url=url,sha256=hashlib.sha256(data).hexdigest())
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:added=list(pool.map(one,items))
rows+=added;p.write_text(json.dumps(rows,indent=2));print(json.dumps(added,indent=2))
