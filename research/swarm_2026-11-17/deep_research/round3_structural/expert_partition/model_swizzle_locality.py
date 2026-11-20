import json,hashlib
from pathlib import Path
from ready_queue_model import route_rows
from indexed_swizzle import map_task
P=Path(__file__).resolve().parent;ROOT=P.parents[3];src=ROOT/'swarm_2026-11-17/deep_research/expert_partition/k_preserve_m1024_01/owned_configs.json';table=json.loads(src.read_text())['indexed'];rows=[]
for m in [3012,8192,16384]:
 for kind in ['uniform','mixed','hot8']:
  rr,_=route_rows(m,kind,917+m)
  for stage,N in [('w13',8192),('w2',4096)]:
   c=next(c for lo,hi,c in table[stage] if lo<m*8<=hi);bm,bn,bk=c['block_shape'];experts=[e for e,r in enumerate(rr) for _ in range((len(r)+bm-1)//bm)];nt=N//bn;tasks=len(experts)*nt
   for workers in [20,132]:
    results=[]
    for group in [1,2,4,8]:
     order=[map_task(i,experts,nt,group) for i in range(tasks)];unique=0
     for start in range(0,tasks,workers):unique+=len(set((experts[mi],ni) for mi,ni in order[start:start+workers]))
     eligible=sum(min(group,len(experts)-i) for i in range(0,len(experts),group) if experts[i]==experts[min(i+group,len(experts))-1] and min(group,len(experts)-i)>1)
     results.append({'group':group,'eligible_Mblocks':eligible,'wave_distinct_fullK_weight_tiles':unique,'logical_fullK_weight_tiles':tasks,'wave_duplicate_fraction':1-unique/tasks if tasks else 0})
    rows.append({'m':m,'pattern':kind,'stage':stage,'SM86_configuration':c,'worker_sensitivity':workers,'Mblocks':len(experts),'rows':results})
r={'status':'CPU_TASK_WAVE_LOCALITY_MODEL','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'scope':'Pinned SM86 tile geometry.132-worker sensitivity is not nativeH200 geometry or performance. Each wave assumed lockstep acrossK; no realcache/bandwidth modeling.','rows':rows,'interpretation':['M/N reorder leaves total logical B/scales loads and dequant arithmetic identical.','Model counts unique(expert,Ntile) per simultaneous taskwave; duplicates may share L2 lines but no hardware multicast guarantee.','20worker SM86 has fewer Ntile tasks perwave than132worker Hopper; localgain may be much larger than nativeHopper transfer.','Groups crossing expertboundaries remain unswizzled; uniformM3012 has few usefulMtiledgroups.','Measure warm/cold fullhelper and profile dequant/ALU/Tensor stalls before attribution.']};(P/'SWIZZLE_LOCALITY_CPU_MODEL.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps([{'m':r['m'],'pattern':r['pattern'],'workers':r['worker_sensitivity'],'stage':r['stage'],'wave_distinct':[v['wave_distinct_fullK_weight_tiles'] for v in r['rows']]} for r in rows if r['stage']=='w13']))
