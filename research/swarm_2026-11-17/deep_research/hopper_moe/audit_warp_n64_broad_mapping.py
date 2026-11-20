from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
j=json.loads((P/'warp_n64_broad_jobs.json').read_text())['jobs'];counts={'output_address_union_checks':0,'full_output_bijections':0,'K_partition_orders':0}
def addresses(bm,wn,n,lane,smem):
 sub=(lane%4)*2+(lane%8)//4;count=64//wn
 return {(8*c+sub)*32+(((n%count*(8//count)+r)^((sub+smem)%8))*4+lane//8)+(n//count)*(bm*32) for c in range(bm//8) for r in range(wn//8)}
for job in j:
 c=job['configuration'];bm,bn,bk=c['block_shape'];wk=c['warp_shape'][2]
 for smem in range(8):
  for n in range(4):
   for lane in range(32):
    a=addresses(bm,32,2*n,lane,smem);b=addresses(bm,32,2*n+1,lane,smem);d=addresses(bm,64,n,lane,smem)
    assert not a&b and a|b==d;counts['output_address_union_checks']+=1
  alladdr=[x for n in range(4) for lane in range(32) for x in addresses(bm,64,n,lane,smem)]
  assert len(alladdr)==bm*bn//2 and len(set(alladdr))==len(alladdr) and set(alladdr)==set(range(bm*bn//2));counts['full_output_bijections']+=1
 for part in range(bk//wk):
  a=[base+part*wk+i*16 for base in range(0,c['shape_k'],bk) for i in range(wk//16)]
  b=[base+part*wk+i*16 for base in range(0,c['shape_k'],bk) for i in range(wk//16)]
  assert a==b;counts['K_partition_orders']+=1
r={'status':'source_address_bijection_passed','checks':counts,'jobs':[x['job_id'] for x in j],'limits':'Checks exact shared-output addresses and partition K order; original B/scale union oracle applies because blockN256/warpK64 unchanged. Does not model shuffles, native register fragment ordering, floating arithmetic or actual hardware. H200 bitwise gates still required.','manifest_sha256':hashlib.sha256((P/'warp_n64_broad_jobs.json').read_bytes()).hexdigest()}
(P/'warp_n64_broad_mapping_results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
