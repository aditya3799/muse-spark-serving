"""CPU contracts for actual direct-global microprobe addresses, fragment ownership and K-bank sequence."""
from pathlib import Path
import ast,hashlib,json
P=Path(__file__).resolve().parent
src=P.parent/'aux_speculation/test_packed_addresses.py'
ns={};tree=ast.parse(src.read_text(encoding='utf-8-sig'));exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef)],type_ignores=[]),str(src),'exec'),ns)
checks=0
for n,k in [(256,128),(256,256)]:
 ref=ns['reference'](n,k,4,True)
 for physical,(r,c)in ref.items():
  lane=(r%8)*4+(c%8)//2;i3=(r%64)//16;a=(c//8)%2;b=(r//8)%2
  inner=((i3*2+a)*2+b)*2+c%2;sub=(inner%8)//2+4*(inner%2)
  word=(c//16)*(n*2)+(r//64)*128+lane*4+inner//8
  assert word*8+sub==physical;checks+=1
ownership=0
for swap in [False,True]:
 for nbase in range(0,256,16 if swap else 8):
  seen=set()
  for lane in range(32):
   for t in range(3 if swap else 2):
    for r in range(4):
     row=t*8+(lane%4)*2+r%2 if swap else t*16+lane//4+(r//2)*8
     n=nbase+lane//4+(r//2)*8 if swap else nbase+(lane%4)*2+r%2
     if row>=24:continue
     assert(row,n)not in seen;seen.add((row,n));ownership+=1
  assert seen==set((r,n)for r in range(24)for n in range(nbase,nbase+(16 if swap else 8)))
fragments=0
# Explicit PTX BF16 m16n8k16 ownership for the source's four A and two B words.
for lane in range(32):
 for reg in range(4):
  for half in range(2):
   a=(lane//4+(reg%2)*8,(lane%4)*2+(reg//2)*8+half)
   assert 0<=a[0]<16 and 0<=a[1]<16;fragments+=1
 for reg in range(2):
  for half in range(2):
   b=((lane%4)*2+reg*8+half,lane//4)
   assert 0<=b[0]<16 and 0<=b[1]<8;fragments+=1
k_cases=[]
for k in [128,4096]:
 for banks in [1,2]:
  seq=[[outer+64*b+16*i+j for outer in range(0,k,64*banks)for i in range(4)for j in range(16)]for b in range(banks)]
  assert sorted(sum(seq,[]))==list(range(k))
  assert all(x==sorted(x)for x in seq)
  # Original BK(64*banks), WK64 bank assignment.
  assert all(all((q//64)%banks==b for q in s)for b,s in enumerate(seq))
  k_cases.append(dict(k=k,banks=banks,sha256=hashlib.sha256(json.dumps(seq).encode()).hexdigest()))
routes=0;negative=0
for count in range(25):
 slots=[8*i+(i%8)for i in range(count)]+[-1]*(24-count)
 assert len(set(x for x in slots if x>=0))==count
 for tile in [8,16]:
  for t in range((24+tile-1)//tile):
   skip=t>0 and slots[t*tile]<0
   assert not skip or all(x<0 for x in slots[t*tile:min(24,(t+1)*tile)])
   routes+=1
bad=[-1]*24;bad[9]=72
assert bad[8]<0 and bad[9]>=0;negative+=1
bad=[0,0]+[-1]*22;assert len(set(x for x in bad if x>=0))<2;negative+=1
r=dict(status='CPU_MAPPING_CONTRACT_PASS_NOT_RUNTIME',packed_address_checks=checks,output_ownership_checks=ownership,fragment_coordinates=fragments,k_sequences=k_cases,frontpacked_tile_checks=routes,negative_controls=negative,scope='Source-literal coordinate and ownership model. No hardware MMA arithmetic result; route frontpacking and unique slots are fixture contracts, not host-inspected at graph replay.',sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in [P/'swap_ab_probe.cuh',P/'swap_ab_probe.py',Path(__file__),src]})
(P/'SWAP_AB_CPU_CONTRACT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k not in ('sources','k_sequences')}))
