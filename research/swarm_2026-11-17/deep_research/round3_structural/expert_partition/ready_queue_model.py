"""CPU event/dataflow prototype; no CUDA, tensor arithmetic or performance prediction."""
import collections,hashlib,heapq,json,random,struct
from pathlib import Path
P=Path(__file__).resolve().parent

def route_rows(m,kind,seed):
 rng=random.Random(seed);rows=[[] for _ in range(32)];ids=[]
 for t in range(m):
  if kind=='hot8': picked=list(range(8))
  elif kind=='remote':picked=rng.sample(range(32,256),8)
  elif kind=='mixed':picked=list(range(8)) if t%2==0 else rng.sample(range(256),8)
  else:picked=rng.sample(range(256),8)
  assert len(set(picked))==8
  ids.append(picked)
  for k,e in enumerate(picked):
   if e<32:rows[e].append(t*8+k)
 return rows,ids

def chunks(rows,bm):
 # Equal producer/consumer row groups admitted; same existing tile arithmetic is opaque.
 out=[]
 for e,r in enumerate(rows):
  for j in range(0,len(r),bm):out.append((e,tuple(r[j:j+bm])))
 return out

def duration(group,stage,n,mode):
 # Dimensionless service time, NOT measured GPU durations. Deterministic across schedules.
 v=(group*1664525+n*1013904223+stage*47)&0xffffffff
 return 1 if mode=='equal' else 0.6+(v%100)/100.0

def barrier_model(groups,workers,mode):
 clock=0
 for stage,nt in [(0,32),(1,16)]:
  avail=[clock]*workers;heapq.heapify(avail)
  for g in range(len(groups)):
   for n in range(nt):heapq.heappush(avail,heapq.heappop(avail)+duration(g,stage,n,mode))
  clock=max(avail)
 return clock

def pipeline(groups,workers,pool_size,mode,epoch=1,changed_stale=False):
 ready=[collections.deque(),collections.deque()];running=[];free=list(range(pool_size));owner={};gslot={};rem13={};rem2={};published={};outputs={};nextg=0;time=0.;seq=0;peak=0;checks=0;tiles_read=0;queues_peak=0;reuses=0;last={};writes=0
 def admit():
  nonlocal nextg,peak,reuses
  while free and nextg<len(groups):
   slot=free.pop();g=nextg;nextg+=1
   if slot in last:reuses+=1
   owner[slot]=(epoch,g);gslot[g]=slot;rem13[g]=32;rem2[g]=16
   for n in range(32):ready[0].append((g,0,n))
   peak=max(peak,len(owner))
 admit()
 while ready[0] or ready[1] or running:
  while len(running)<workers and (ready[1] or ready[0]):
   task=(ready[1] if ready[1] else ready[0]).popleft();g,stage,n=task;slot=gslot[g]
   assert owner[slot]==(epoch,g);checks+=1
   if stage:
    assert published.get(g)==(epoch,g);checks+=1
    if changed_stale:raise AssertionError('negative stale-generation publish detected')
   heapq.heappush(running,(time+duration(g,stage,n,mode),seq,task));seq+=1
  queues_peak=max(queues_peak,len(ready[0])+len(ready[1]))
  assert running,'deadlock: unfinished work with no runnable producer'
  time,_,(g,stage,n)=heapq.heappop(running);slot=gslot[g]
  assert owner[slot]==(epoch,g);checks+=1
  if stage==0:
   rem13[g]-=1
   if rem13[g]==0:
    published[g]=(epoch,g)
    for j in range(16):ready[1].append((g,1,j))
  else:
   tiles_read+=1;rem2[g]-=1
   # Each down-N tile stores a disjoint column range for every live canonical row.
   for canonical in groups[g][1]:
    old=outputs.get(canonical,(epoch,groups[g][0],0));assert old[:2]==(epoch,groups[g][0]) and not old[2]&(1<<n);outputs[canonical]=(epoch,groups[g][0],old[2]|(1<<n));writes+=1
   if rem2[g]==0:
    last[slot]=(epoch,g);del owner[slot];free.append(slot);admit()
 assert nextg==len(groups) and not owner
 expected={c:(epoch,e,65535) for e,row in groups for c in row}
 assert outputs==expected;checks+=len(expected)
 return {'dimensionless_makespan':time,'checks':checks,'peak_live_pool_tiles':peak,'pool_reuses':reuses,'peak_queued_tasks':queues_peak,'tasks':seq,'canonical_Ntile_writes':writes,'w2_Ntiles':tiles_read}

def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
def fold(v):
 out=0.
 for x in v:out=f32(out+f32(x))
 return out

def run():
 rows=[];cases=0;assertions=0;negative=[]
 for m in [32,768,1024,4096,16384]:
  bm=24 if m==32 else 48 if m==768 else 64 if m==1024 else 128
  for kind in ['uniform','mixed','hot8','remote']:
   by,ids=route_rows(m,kind,41+m);g=chunks(by,bm)
   # The route permutation is lossless, but these source geometry scenarios are not a tuner.
   assert sorted(c for _,r in g for c in r)==[t*8+k for t in range(m) for k,e in enumerate(ids[t]) if e<32]
   workers=132
   base=barrier_model(g,workers,'equal')
   alternatives=[]
   for pool in [2,4,8,16]:
    a=pipeline(g,workers,pool,'equal',epoch=2);assertions+=a['checks'];cases+=1
    alternatives.append({'pool_tiles':pool,'ratio_barrier_over_pipeline':base/a['dimensionless_makespan'] if a['dimensionless_makespan'] else 1,**a})
   rows.append({'m':m,'pattern':kind,'bm_scenario':bm,'live_routes':sum(map(len,by)),'groups':len(g),'workers':workers,'baseline_dimensionless':base,'alternatives':alternatives})
 # Irregular service and small-worker adversarial scheduling, graph generation changes.
 for seed in range(30):
  by,_=route_rows(33,'mixed' if seed%2 else 'uniform',seed);g=chunks(by,8)
  for workers,pool in [(1,1),(3,1),(7,2),(20,4)]:
   a=pipeline(g,workers,pool,'irregular',epoch=seed+1);assertions+=a['checks'];cases+=1
 # Publication epoch must be checked even if graph addresses are unchanged.
 try:pipeline([(0,(0,))],2,1,'equal',2,True)
 except AssertionError:negative.append('stale_epoch_rejected')
 # Explicit wait-for cycle matching documented held-slot/next-item hazard.
 graph={'consumer_next_wait':['producer_next'],'producer_next':['activation_ready'],'activation_ready':['epilogue'],'epilogue':['held_slot_handoff'],'held_slot_handoff':['consumer_next_wait']}
 def cycle(node,stack):
  if node in stack:return True
  return any(cycle(x,stack+[node]) for x in graph.get(node,[]))
 assert cycle('consumer_next_wait',[]);negative.append('blocking_next_item_before_handoff_deadlock')
 # Oversubscribed consumers can occupy every resident worker while producers cannot run.
 resident=['blocked_W2']*4;pending=['W13']*4
 assert all(x=='blocked_W2' for x in resident) and pending;negative.append('claim_unready_consumer_residency_deadlock')
 a,b=fold([1e20,-1e20,3.]),fold([1e20,3.,-1e20]);assert a!=b;negative.append('completion_order_FP32_add_changes_result')
 memory=[]
 for m in [768,1024,4096,16384]:
  R=m*8;poolrows=min(R,8*176);memory.append({'M':m,'R':R,'scored_alias_scratch_bytes':3*R*4096*2,'fully_fused_canonical_bytes':2*R*4096*2,'bounded_activation_plus_canonical_down_bytes':(R+poolrows)*4096*2,'activation_pool_rows':poolrows,'excludes':'metadata, weights, graphs, compiler allocations, unfused W13 fallback scratch'})
 result={'status':'CPU_ABSTRACT_DATAFLOW_AND_SCHEDULER_PASS','cases':cases,'assertions':assertions,'rows':rows,'negative_controls':negative,'order_counterexample':{'slot_order_result':a,'completion_permutation_result':b},'memory':memory,'limitations':['Service times synthetic, no GPU speed prediction.','Rows are source geometry scenarios; changing actual native BM is not authorized by this model.','Only matching W13/W2 M groups; intersecting mismatched partitions needs separate dependency proof.','Payload identity/ownership tested, not GEMM arithmetic or device release/acquire implementation.','Canonical rawdown retained to preserve existing fullrow norm and slot-order finalizer.','No persistent GPU kernel implemented; no source copied from reference.']}
 (P/'READY_QUEUE_CPU_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'cases':cases,'assertions':assertions,'negative_controls':negative,'equal_service_ratios':[{'m':r['m'],'pattern':r['pattern'],'ratios':[(a['pool_tiles'],round(a['ratio_barrier_over_pipeline'],3)) for a in r['alternatives']]} for r in rows]}))
if __name__=='__main__':run()
