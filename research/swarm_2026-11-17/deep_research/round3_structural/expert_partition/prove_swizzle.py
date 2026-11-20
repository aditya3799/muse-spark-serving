from pathlib import Path
import json,random,hashlib
from indexed_swizzle import map_task
P=Path(__file__).resolve().parent;cases=[];checks=0;rng=random.Random(91)
for ne in [1,2,7,32]:
 for seed in range(30):
  counts=[rng.randrange(0,65) for _ in range(ne)];experts=[e for e,c in enumerate(counts) for _ in range(c)]
  for n in [1,16,32,64,128]:
   for group in [1,2,4,8]:
    mapped=[map_task(i,experts,n,group) for i in range(len(experts)*n)]
    assert sorted(mapped)==[(m,j) for m in range(len(experts)) for j in range(n)];checks+=len(mapped)
    for i,(m,j) in enumerate(mapped):
     assert experts[m]==experts[i//n];checks+=1
    # Persistent CTA-strided traversal collectively covers all tasks exactly once.
    for workers in [1,20,132]:
     parts=[mapped[cta::workers] for cta in range(workers)];assert sum(map(len,parts))==len(mapped);checks+=1
    if seed==0:cases.append({'experts':ne,'m_blocks':len(experts),'Nblocks':n,'group':group,'swizzled_tasks':sum(pair!=(i//n,i%n) for i,pair in enumerate(mapped))})
r={'status':'CPU_BIJECTION_EXPERT_BOUNDARY_AND_PERSISTENT_STRIDE_PASS','configurations':120*5*4,'checks':checks,'sample_cases':cases,'source_sha256':hashlib.sha256((P/'indexed_swizzle.py').read_bytes()).hexdigest(),'limits':['Only aligned tile indices; GPU row-metadata ordering and device execution gates remain.','Grouped scheduler deliberately excluded because expert cursor relies on monotonic M.','No math changes or extra GPU metadata kernel; two expert endpoint loads per tile.','No locality or performance claim from bijection.']};(P/'INDEXED_SWIZZLE_CPU_PROOF.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ['status','configurations','checks']}))
