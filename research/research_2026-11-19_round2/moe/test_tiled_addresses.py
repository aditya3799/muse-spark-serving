import hashlib,json,time
from pathlib import Path
import numpy as np
here=Path(__file__).resolve().parent;started=time.time();rows=[]
def address(n,k,width,up):
 p=n%4096//8*16+n%8+n//4096*8 if up else n
 tile=((p//256)*16+k//256)*4+p%256//64
 ix=k%256//64*512+p%64//16*128+p%8*16+k%64//16*4+k%8//2
 return tile*2304+ix,p//8%2+(k//8%2)*2
for width in (4096,8192):
 checked=0;up=width==8192
 for first in range(0,width*512,65536):
  z=np.arange(first,min(first+65536,width*512),dtype=np.int64)
  kb=z//(width*2);nb=z%(width*2)//128;lane=z%128//4;a=z%4
  n=nb*64+a*16+lane//4;k=kb*16+lane%4*2
  for i in range(2):
   reused,_=address(n+(8*i if up else 0),k,width,up)
   for j in range(2):
    parity=n//4096 if up else i
    for h in range(2):
     exact,r=address(n+8*i,k+8*j+h,width,up)
     assert np.array_equal(reused,exact)
     assert np.all(r==parity+2*j)
     checked+=len(z)
 rows.append(dict(width=width,checked_coefficients=checked,source_word_reuse_exact=True))
report=dict(complete=True,rows=rows,seconds=time.time()-started,tiled_sha256=hashlib.sha256((here/'wgs_bridge_tiled.py').read_bytes()).hexdigest(),scope='Exhaustive fixed geometry address/r selector proof; independent forward packer GPU gate still required')
(here/'bridge_tiled_cpu.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
