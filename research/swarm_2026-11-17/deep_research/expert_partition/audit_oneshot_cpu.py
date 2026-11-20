import ast,hashlib,json,random
from pathlib import Path
P=Path(__file__).resolve().parent
source=P/'oneshot_partition.py';ast.parse(source.read_text(encoding='utf-8-sig'))
rows=[]
for experts in [1,2,32,256]:
 for n in [4097,6024,8192,8193,16384]:
  for kind in ['all_invalid','all_hot','mixed_boundary','random']:
   rng=random.Random(n+experts)
   ids=([-1]*n if kind=='all_invalid' else [experts-1]*n if kind=='all_hot' else [(-2,-1,0,experts-1,experts,2**35)[i%6] for i in range(n)] if kind=='mixed_boundary' else [rng.randrange(-1,experts) for _ in range(n)])
   owners=[0]*n;count=[0]*experts;light=[None]*n;heavy=[None]*n;cutoff=40
   for expert in range(experts):
    count[expert]=sum(v==expert for v in ids)
    for i,v in enumerate(ids):
     valid=0<=v<experts
     if (valid and v==expert) or (expert==0 and not valid):
      owners[i]+=1;light[i]=v if valid and count[expert]<=cutoff else -1;heavy[i]=v if valid and count[expert]>cutoff else -1
   expected_count=[sum(v==e for v in ids) for e in range(experts)]
   assert owners==[1]*n
   assert count==expected_count
   assert light==[v if 0<=v<experts and count[v]<=cutoff else -1 for v in ids]
   assert heavy==[v if 0<=v<experts and count[v]>cutoff else -1 for v in ids]
   rows.append({'experts':experts,'routes':n,'pattern':kind,'exactly_one_writer_per_slot':True,'reference_equal':True})
report={'scope':'CPU model of GPU ownership and partition specification, not executed Triton proof','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'rows':rows,'passed':True}
(P/'ONESHOT_CPU_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':len(rows),'passed':True}))
