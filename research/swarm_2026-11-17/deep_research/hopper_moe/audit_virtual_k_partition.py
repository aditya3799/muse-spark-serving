"""CPU arithmetic-tree and virtual K-partition address oracle, not a kernel."""
from pathlib import Path
import json,struct,random,hashlib
P=Path(__file__).resolve().parent;OLD=P.parents[2]/'swarm_2026-11-16/moe/phase4_hour'
jobs=json.loads((OLD/'producer_compile_jobs.json').read_text())['jobs'];rng=random.Random(121)
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
def tree(values):
 v=list(values);i=1
 while i<len(v):
  for p in range(len(v)//(2*i)):v[p]=f32(v[p]+v[p+len(v)//(2*i)])
  i*=2
 return v[0]
def serial_wrong(values):
 result=values[0]
 for x in values[1:]:result=f32(result+x)
 return result
checks={'B_byte_offsets':0,'scale_byte_offsets':0,'A_descriptor_offsets':0,'K_partition_sequences':0,'FP32_reduction_trees':0};rows=[]
for m in (8,32,128):
 for stage in ('w13','w2'):
  j=next(j for j in jobs if j['stage']==stage and any(lo<m*8<=hi for lo,hi in j['intervals']));c=j['baseline'];bm,bn,bk=c['block_shape'];wm,wn,wk=c['warp_shape'];parts=bk//wk
  if parts==1:continue
  assert wk==64 and wn==32 and parts in (2,4)
  originalthreads=(bn//wn)*parts*32;newthreads=originalthreads//parts
  for part in range(parts):
   for it in range(4):
    vit=4*part+it
    oldA=part*bm*128+it*32;newA=(vit//4)*bm*128+(vit%4)*32
    assert oldA==newA;checks['A_descriptor_offsets']+=1
    for nw in range(bn//wn):
     for lane in range(32):
      idx=(bn//wn//2)*32*4*part+32*(nw//2)+lane
      oldB=it*(bn*16//2)+(idx*2+nw%2)*8
      newidx=32*(nw//2)+lane
      newB=vit*(bn*16//2)+(newidx*2+nw%2)*8
      assert oldB==newB;checks['B_byte_offsets']+=1
      oldS=(part*64+it*16)//16*bn+(lane//4*2+nw%2+(nw//2)*16)*4
      newS=vit*bn+(lane//4*2+nw%2+(nw//2)*16)*4
      assert oldS==newS;checks['scale_byte_offsets']+=1
   oldK=[base+part*64+it*16 for base in range(0,4096,bk) for it in range(4)]
   newK=[base+(vit//4)*64+(vit%4)*16 for base in range(0,4096,bk) for vit in range(parts*4) if vit//4==part]
   assert oldK==newK;checks['K_partition_sequences']+=1
  for _ in range(2048):
   vals=[f32(rng.uniform(-1,1)*2**rng.randrange(-30,31)) for _ in range(parts)]
   reference=f32(vals[0]+vals[1]) if parts==2 else f32(f32(vals[0]+vals[2])+f32(vals[1]+vals[3]))
   assert struct.pack('<f',tree(vals))==struct.pack('<f',reference);checks['FP32_reduction_trees']+=1
  rows.append({'tokens':m,'stage':stage,'block':c['block_shape'],'warp_K_kept':64,'virtual_partitions':parts,'original_threads':originalthreads,'proposed_threads':newthreads,'FP32_output_accumulator_regs_per_thread':wm*parts,'shared_reduction_syncs_removed_model':max((wm+15)//16,1)*2*(parts.bit_length()-1),'status':'Concept only; no C++ prototype or H200 execution'})
negative=[f32(1e20),f32(1),f32(-1e20),f32(1)];assert tree(negative)==2 and serial_wrong(negative)==1
r={'status':'CPU_concept_model_passed','checks':checks,'counterexample':{'partial_sums':negative,'correct_tree':tree(negative),'wrong_sequential_sum':serial_wrong(negative)},'rows':rows,'strict_constraints':['Keep each original K64 partition in independent FP32 accumulator bank until final original tree reduction','Native WGMMA primitive, BF16 conversion, scale placement and final BF16 cast must remain unchanged','Do not enlarge WarpShapeK because native A swizzle asserts<=64; custom virtual-part descriptors required','Do not use current shared reducer after collapsing physical Kwarps; explicit bank tree reduction required','No gain assumed: serialized groups can reduce tensor-core parallelism and increase per-thread register pressure'],'limits':'Integer addresses and abstract FP32tree only; no native mapping, shuffle, hardware race, correctness or performance proof.'}
(P/'VIRTUAL_K_PARTITION_CONCEPT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'checks':checks,'rows':rows,'negative':r['counterexample']},indent=2))
