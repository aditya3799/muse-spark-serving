import hashlib,json,subprocess
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1];pin='297e54fd105e974cdf401e5b4d20253a22c16f90'
out=here/'dflash_sources';out.mkdir(exist_ok=True);records=[]
for name in ['speculative/dflash_worker_v2.py','speculative/dflash_utils.py','speculative/draft_worker_common.py','models/dflash.py']:
 path='python/sglang/srt/'+name
 data=subprocess.check_output(['git','-C',str(root/'sglang'),'show',pin+':'+path])
 (out/Path(name).name).write_bytes(data)
 records.append(dict(path=path,pin=pin,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
(out/'manifest.json').write_text(json.dumps(records,indent=2))
print('snapshotted',len(records),'frozen upstream files; release patch does not alter them')
