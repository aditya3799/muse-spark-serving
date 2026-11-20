"""Fetch only pinned public CUTLASS include files; stream archive without storing it."""
from pathlib import Path
import urllib.request,tarfile,json,hashlib,time
here=Path(__file__).resolve().parent
commit='cd9d8a7d612b6b9ad0f1b698925e31243e828629'
url='https://codeload.github.com/NVIDIA/cutlass/tar.gz/'+commit
base=here/'cutlass_pinned';base.mkdir(exist_ok=True)
records=[];total=0;start=time.time()
with urllib.request.urlopen(url,timeout=120) as stream:
 with tarfile.open(fileobj=stream,mode='r|gz') as archive:
  for member in archive:
   parts=Path(member.name).parts
   if len(parts)<3 or parts[1]!='include' or not member.isfile():continue
   rel=Path(*parts[1:]);assert '..' not in rel.parts
   total+=member.size
   if total>120*1024**2:raise RuntimeError('include download exceeds 120 MiB budget')
   data=archive.extractfile(member).read();assert len(data)==member.size
   target=base/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
   records.append(dict(path=rel.as_posix(),size=len(data),sha256=hashlib.sha256(data).hexdigest()))
(base/'manifest.json').write_text(json.dumps(dict(commit=commit,url=url,bytes=total,files=records,elapsed=time.time()-start),indent=2))
print('Pinned CUTLASS includes:',len(records),'files',total,'bytes',round(time.time()-start,2),'seconds')
p=here/'build_host.py';s=p.read_text().replace('include=Path("/tmp/okc-cutlass-cd9d8a7d/include")','include=d/"cutlass_pinned/include"');p.write_text(s)
