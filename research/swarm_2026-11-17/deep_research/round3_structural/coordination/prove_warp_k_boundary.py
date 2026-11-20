"""Distinguish32-wide and64-wide K-bank accumulation on exact BF16 inputs."""
import struct,json,hashlib
from pathlib import Path
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
def run(k,wk,products):
 banks=[];coverage=[];values=[]
 for bank in range(2):
  seq=[outer+bank*wk+inner for outer in range(0,k,2*wk) for inner in range(wk)]
  coverage+=seq;v=f32(0.)
  for pos in seq:v=f32(v+products.get(pos,0.))
  banks.append(dict(bank=bank,first_positions=seq[:min(48,len(seq))],sequence_sha256=hashlib.sha256(json.dumps(seq).encode()).hexdigest(),sum=v));values.append(v)
 assert sorted(coverage)==list(range(k))
 return dict(k=k,bk=2*wk,wk=wk,banks=banks,raw=f32(values[0]+values[1]))
products={0:float(2**24),32:1.,64:float(-(2**24)),96:1.}
for v in products.values():assert struct.unpack('<I',struct.pack('<f',v))[0]&65535==0
cases=[]
for k in (128,256,4096):
 a=run(k,32,products);b=run(k,64,products)
 assert a['raw']==2. and b['raw']==1.
 a['after_BF16_then_x4']=8.;b['after_BF16_then_x4']=4.
 cases.append(dict(required_WK32=a,wrong_WK64=b))
r=dict(status='CPU_EXACT_BANK_ASSIGNMENT_NEGATIVE_CONTROL_PASS',cases=cases,products=products,
 operand_realization='NVFP4 q=4 with group-scale1 yields exact quarterweight1. Activations2^24,1,-2^24,1 are exactly BF16-representable; allotherK activationszero.',
 scope='The symbolic bank-sum model intentionally isolates source K-bank assignment. A native MMA fixture must still verify the products, instruction grouping and pinned reducer.',
 consequence='Keeping two K banks is insufficient: BK/WK and strided membership must match reference. A localWK32 candidate cannot silently reuseWK64 sequencing.')
Path(__file__).with_name('WARP_K_BOUNDARY_NEGATIVE_CONTROL.json').write_text(json.dumps(r,indent=2)+'\n')
print('3 K extents distinguish raw2/x4=8 from wrong raw1/x4=4')
