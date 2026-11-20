"""Adversarial event model of the unchanged three-slot WGS rings plus split ready.

This is a protocol/order check, NOT a GPU timing model. Four warpgroup actors
represent the16 consumer-warp arrivals (four arrivals per actor). Independent TMA
completions and delayed phase-B inputs stress overwrite and readiness ordering.
"""
import heapq,json,random,time
from pathlib import Path

def simulate(seed,split,*,slots=3,count=128,bug=None):
 rng=random.Random(seed);groups=4
 wr=[None]*slots;xr=[None]*slots
 wp=[0]*slots;xp=[0]*slots;jp=[0]*slots;jlast=[None]*slots
 we=[0]*slots;xe=[0]*slots
 wa=[set(range(groups)) for _ in range(slots)];xa=[set(range(groups)) for _ in range(slots)]
 consumers=[dict(s=0,phase='begin',mma=0,previous=None) for _ in range(groups)]
 pending=[];issued=[0,0];tick=0;early=0;checks=0;events=0
 def release(acks,seq,g):
  slot=seq%slots
  assert g not in acks[slot],('duplicate release',seq,g)
  acks[slot].add(g)
  if len(acks[slot])==groups:
   phase=we if acks is wa else xe
   phase[slot]^=1
 while not all(c['s']==count for c in consumers):
  tick+=1
  if tick>200000:raise RuntimeError('event-model deadlock')
  while pending and pending[0][0]<=tick:
   _,which,seq=heapq.heappop(pending)
   target=wr if which==0 else xr
   target[seq%slots]=seq
   phase=wp if which==0 else xp
   phase[seq%slots]^=1
   if wr[seq%slots]==xr[seq%slots]==seq and jlast[seq%slots]!=seq:
    jp[seq%slots]^=1;jlast[seq%slots]=seq
  actor=rng.randrange(groups+2)
  if actor<2:
   which=actor;seq=issued[which]
   if seq>=count:continue
   if which==1 and seq>=count//2 and any(c['s']<count//2 for c in consumers):continue
   acks=wa if which==0 else xa;target=wr if which==0 else xr
   slot=seq%slots
   if seq>=slots:
    phase=we if which==0 else xe
    empty_ready=phase[slot]!=(((seq//slots)&1)^1)
    assert empty_ready==(len(acks[slot])==groups),('empty barrier stale parity',seed,which,seq)
    if not empty_ready:continue
   elif len(acks[slot])!=groups:continue
   acks[slot]=set();target[slot]=None;issued[which]+=1
   heapq.heappush(pending,(tick+rng.randrange(1,80 if which else 45),which,seq));events+=1
   continue
  g=actor-2;c=consumers[g];seq=c['s']
  if seq==count:continue
  slot=seq%slots
  if c['phase']=='begin':
   phase=wp if split else jp
   ready=phase[slot]!=((seq//slots)&1)
   if not ready:continue
   assert wr[slot]==seq and (split or xr[slot]==seq),('full barrier stale parity',seed,seq)
   
   # The first coefficient group may be decoded before input readiness.
   if xr[slot]!=seq:early+=1
   c['phase']='first_mma';checks+=1
  elif c['phase']=='first_mma':
   if bug!='missing_input_wait':
    ready=xp[slot]!=((seq//slots)&1)
    if not ready:continue
   assert xr[slot]==seq,('MMA before activation completion',seed,seq,g)
   assert wr[slot]==seq,('coefficient read before weight completion',seed,seq,g)
   if c['previous'] is not None:
    release(xa,c['previous'],g);c['previous']=None
   c['mma']=1;c['phase']='mma';checks+=2
  else:
   assert xr[slot]==seq,('activation overwritten during MMA',seed,seq,g)
   if c['mma']<6:assert wr[slot]==seq,('weights overwritten before register loading completed',seed,seq,g)
   c['mma']+=1;checks+=2
   if c['mma']==6:release(wa,seq,g)
   if c['mma']==8:
    if (seq+1)%16==0 or seq+1==count:release(xa,seq,g)
    else:c['previous']=seq
    c['s']+=1;c['phase']='begin'
 assert issued==[count,count]
 assert all(len(a)==groups for a in wa+xa)
 return dict(seed=seed,split=split,slots=slots,stages=count,parity_wraps_per_slot=count//slots//2,readiness_checks=checks,headstart_occurrences=early,tma_events=events)

started=time.time();cases=[]
for seed in range(930401,930465):
 for split in (False,True):cases.append(simulate(seed,split))
negative=[]
for seed in range(930401,930409):
 try:simulate(seed,True,bug='missing_input_wait')
 except AssertionError:negative.append(seed)
 else:raise AssertionError(('invalid model escaped',seed))
# One slot cannot support the incumbent delayed previous-stage activation release.
try:simulate(930401,True,slots=1,count=32)
except RuntimeError:negative.append('one_slot_deadlock')
else:raise AssertionError('invalid one-slot ring escaped')
report=dict(complete=True,cases=len(cases),stages_checked=sum(x['stages']*4 for x in cases),readiness_checks=sum(x['readiness_checks'] for x in cases),split_headstarts=sum(x['headstart_occurrences'] for x in cases if x['split']),joined_headstarts=sum(x['headstart_occurrences'] for x in cases if not x['split']),negative_controls=negative,seconds=time.time()-started,explicit_full_and_empty_mbarrier_parity=True,no_gpu_timing_model=True,scope='Source-order protocol model only; actual SM90 memory ordering, async WGMMA execution and deadlock gate still required',rows=cases)
Path(__file__).with_name('split_ready_cpu_generations.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
