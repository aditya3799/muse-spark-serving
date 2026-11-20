"""Bounded source-only snapshot of the two new decode commits."""
import hashlib,json,urllib.request
from pathlib import Path
here=Path(__file__).resolve().parent;out=here/'pr51393';out.mkdir(exist_ok=True)
rows=[]
for commit in ('c868e043ec4efefb09a2b156509449a9fc5653b7','36c4cd4d2e8fa600bd5d1587c9b6fc5fd903fe59'):
 url='https://github.com/sgl-project/sglang/commit/'+commit+'.patch'
 req=urllib.request.Request(url,headers={'User-Agent':'bounded-source-audit'})
 with urllib.request.urlopen(req,timeout=40) as r:
  data=r.read(2000001)
 assert len(data)<=2000000,'patch exceeded bounded source budget'
 path=out/(commit+'.patch');path.write_bytes(data)
 rows.append(dict(url=url,commit=commit,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
(out/'manifest.json').write_text(json.dumps(dict(pr='https://github.com/sgl-project/sglang/pull/51393',head=rows[-1]['commit'],status_at_audit='open; unmerged',files=rows),indent=2))
print([(r['commit'][:8],r['bytes']) for r in rows])
