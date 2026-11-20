"""CPU proof of indexed multicast expert-boundary hazard; no CUDA claim."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent

def experts_per_block(counts,bm,cluster,cluster_align):
    result=[]
    quantum=bm*(cluster if cluster_align else 1)
    for e,count in enumerate(counts):
        result.extend([e]*(((count+quantum-1)//quantum)*(quantum//bm)))
    return result

def hazards(ids,c):
    return [{'blocks':list(range(i,min(i+c,len(ids)))),'expected_experts':ids[i:i+c],
             'multicast_source_expert':ids[i]} for i in range(0,len(ids),c)
            if len(ids[i:i+c])!=c or any(e!=ids[i] for e in ids[i:i+c])]

cases=[]
for bm in [8,48,128,176]:
 for c in [2,4]:
  for name,counts in [('one_route_each',[1]*32),('uneven',[bm+1,1,3*bm,0,bm-1]),('already_aligned',[c*bm]*8)]:
   original=experts_per_block(counts,bm,c,False);fixed=experts_per_block(counts,bm,c,True)
   bad=hazards(original,c);good=hazards(fixed,c);assert not good
   if name=='one_route_each':assert bad
   if name=='already_aligned':assert not bad
   cases.append({'bm':bm,'cluster':c,'pattern':name,'original_blocks':len(original),'cluster_aligned_blocks':len(fixed),'original_hazards':bad,'aligned_hazards':good})
v=P.parent/'humming_vendor/humming'
files=['kernel/humming.py','include/humming/scheduler.cuh','include/humming/memory/g2s_loader/loader_b.cuh']
r={'status':'PASS_COUNTEREXAMPLE','cases':cases,'checks':len(cases),'source_hashes':{f:hashlib.sha256((v/f).read_bytes()).hexdigest() for f in files},'finding':'Plain multi_cast_size_b>1 is unsafe for indexed expert-aligned routes. Scheduler distributes adjacent BM tiles then derives expert independently; onlyrank0 issues B multicast. Adjacent expert blocks can differ. Oddtotal blocks can also let rank read past initialized expert_ids.','required_change':'Align each expert route sequence to BM*cluster, then expand each aligned expert ID into cluster identical IDs indexed at BM stride. Merelypassing larger alignmentblock without expanding expert_ids is also wrong. Padfinal cluster. Keep sameBM/warp/K arithmetic; test exactmappedinputownership and originalraw/finalgates onHopper.','cost':'Each activeexpert adds up to BM*(cluster-1) paddedslots versusBM alignment; canadd uselessCTAs/clusterresidency andlimits. AtmostoneBMtile/expertcommondecode cannotbenefit. Largeprefill multipletiles/expert mayreduce B transactions, notbydefault2xserving.','decision':'No exportenable orGPUrun; existingproducerpolicy keepsmulti_cast_size_b=1.'}
(P/'multicast_boundary_audit.json').write_bytes((json.dumps(r,indent=2)+'\n').encode())
print('PASS',len(cases),'CPUcases; config-onlymulticastrejected')
