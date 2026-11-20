import json
from pathlib import Path
import torch
from finite_scale_scan import verify_scale_bytes,verify_resident_scale_bytes
here=Path(__file__).resolve().parent
rows=[]
a=torch.arange(256,device='cuda',dtype=torch.int32).to(torch.uint8)
finite=a[(a&127)!=127]
for size in (1,255,256,4095,4096,4097,65537):
 x=finite.repeat((size+len(finite)-1)//len(finite))[:size].contiguous()
 y=x.clone()
 assert verify_scale_bytes(x,y)
 rows.append(dict(size=size,finite=True,exact=True))
 for stage in (x,y):
  for at in sorted(set((0,size//2,size-1))):
   old=stage[at].clone()
   for bad in (127,255):
    stage[at]=bad;assert not verify_scale_bytes(x,y)
    rows.append(dict(size=size,stage=0 if stage is x else 1,offset=at,bad=bad,rejected=True))
   stage[at]=old
 assert verify_scale_bytes(x,y)
# Actual WGS scale-region addressing: arbitrary code bytes including127/255
# must not trigger the finite-SF gate; each packed scale region is scanned.
class Resident:pass
resident=Resident()
for name,n in (('w13',8192),('w2',4096)):
 stage=torch.randint(0,256,(n*2304,),device='cuda',dtype=torch.uint8)
 stage.view(-1,9216)[:,8192:].fill_(56)
 setattr(resident,name,stage)
assert verify_resident_scale_bytes(resident)
for name in ('w13','w2'):
 stage=getattr(resident,name)
 for at in (8192,9215,stage.numel()-1024,stage.numel()-1):
  for bad in (127,255):
   stage[at]=bad;assert not verify_resident_scale_bytes(resident)
   rows.append(dict(packed_stage=name,offset=at,bad=bad,rejected=True))
  stage[at]=56
 assert verify_resident_scale_bytes(resident)
report=dict(complete=True,cases=len(rows),rows=rows,scratch_bytes=4,scope='Actual scale-byte finite scan; no fast-kernel activation or H200 execution')
(here/'finite_scale_scan_gpu.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
