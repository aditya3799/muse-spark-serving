"""Source-model oracle for W13 warpN32 ->64 loader and epilogue ownership.
This checks addresses and K order, not native MMA execution or floating behavior.
"""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
V=P.parents[2]/'swarm_2026-11-16/moe/humming_vendor/humming/include/humming'
def b_bytes(wn,nw,lane,kpart,it):
 half=wn==32;true_n=4;idx=32*(nw//2 if half else nw)+lane+true_n*32*4*kpart
 width=8 if half else 16
 start=(idx*2+nw%2)*width if half else idx*width
 return set(range(it*2048+start,it*2048+start+width))
def scale_bytes(wn,nw,lane,kpart,it):
 width=wn//8;warps=64//wn;pack=64//width;group=kpart*4+it
 idx=(lane//4)*warps+nw%warps+(nw//warps)*pack+group*(256//width)
 return set(range(idx*width,(idx+1)*width))
def epilogue(wn,nw,lane,smem):
 sub=(lane%4)*2+(lane%8)//4;count=64//wn
 return {sub*32+(((nw%count*(8//count)+r)^((sub+smem)%8))*4+lane//8)+(nw//count)*256 for r in range(wn//8)}
checks={'b_byte_unions':0,'scale_byte_unions':0,'epilogue_unions':0,'k_sequences':0}
for n in range(4):
 for lane in range(32):
  for kp in range(2):
   for it in range(4):
    for fn,key in ((b_bytes,'b_byte_unions'),(scale_bytes,'scale_byte_unions')):
     a=fn(32,2*n,lane,kp,it);b=fn(32,2*n+1,lane,kp,it);c=fn(64,n,lane,kp,it)
     assert not a&b and a|b==c,(key,n,lane,kp,it)
     checks[key]+=1
  for smem in range(8):
   a=epilogue(32,2*n,lane,smem);b=epilogue(32,2*n+1,lane,smem);c=epilogue(64,n,lane,smem)
   assert not a&b and a|b==c
   checks['epilogue_unions']+=1
for kp in range(2):
 old=[base+kp*64+it*16 for base in range(0,4096,128) for it in range(4)]
 new=[base+kp*64+it*16 for base in range(0,4096,128) for it in range(4)]
 assert old==new;checks['k_sequences']+=1
sources=['memory/s2r_loader/loader_b.cuh','memory/s2r_loader/loader_bs.cuh','epilogue/smem_writer.cuh','mma/wgmma.cuh']
r={'status':'source_mapping_oracle_passed','checks':checks,'relationship':'Each new warpN64 warp covers the union of two old warpN32 warps at equal lane id, with unchanged per-K-partition slab order. Native instruction expression equality is separately asserted by offline compiler harness.','limitations':['Does not emulate dequant fragment selection, WGMMA lane collectives, shuffles or floating accumulation. Actual H200 bitwise gate remains required.','W2 blockN128 excluded because warpN64 would give N_WARPS2, violating WGMMA 4-warp groups.'],'source_sha256':{s:hashlib.sha256((V/s).read_bytes()).hexdigest() for s in sources}}
(P/'warp_n64_mapping_results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
