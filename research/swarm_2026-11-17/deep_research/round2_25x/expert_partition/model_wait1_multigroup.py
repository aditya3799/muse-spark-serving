"""Independent adversarial multi-warpgroup schedule model; CPU only."""
import json,pathlib,random
P=pathlib.Path(__file__).resolve().parent

def simulate(stages,tiles,groups,seed,barrier=True,drain=True):
 rng=random.Random(seed);positions=[0]*groups;pending=[[] for _ in range(groups)];blocked={};done=set();checks=0
 def check_write(stage):
  nonlocal checks
  assert all(g['stage']!=stage for queue in pending for g in queue),('shared_A_overwrite',stage,pending)
  checks+=1
 while len(done)<groups:
  choices=[g for g in range(groups) if g not in done and g not in blocked]
  if not choices:
   assert len(blocked)==groups and len(set(blocked.values()))==1,('nonuniform barrier',blocked)
   tile=next(iter(blocked.values()));stage=tile%stages
   if tiles-tile>=stages:check_write((stage+stages-1)%stages)
   for group in list(blocked):
    assert all(v['bank']!=1 for v in pending[group]),'postbarrier transform bank1 conflict'
    positions[group]+=1
   blocked.clear();continue
  group=rng.choice(choices);position=positions[group]
  if position==tiles*4:
   if drain:pending[group].clear()
   done.add(group);continue
  tile,i=divmod(position,4);stage=tile%stages;bank=i%2
  # qbits/scales prefetch uses separate storage; no regs_b mutation yet.
  pending[group].append({'tile':tile,'stage':stage,'bank':bank})
  while len(pending[group])>1:pending[group].pop(0)
  if i==2:
   if barrier:blocked[group]=tile;continue
   if tiles-tile>=stages:check_write((stage+stages-1)%stages)
  assert all(v['bank']!=(i+1)%2 for v in pending[group]),'register operand overwrite'
  checks+=1;positions[group]+=1
 assert all(not q for q in pending),('final drain missing',pending)
 return checks
positive=[]
for stages in (3,4):
 for tiles in (1,2,3,4,5,7,8,9,15,16,17,31,32,33,63,64,65):
  for groups in (1,2,4):
   for seed in range(10):positive.append(simulate(stages,tiles,groups,seed))
negative=[]
for mutation,kwargs in (('missing_final_drain',{'drain':False}),('removed_producer_barrier',{'barrier':False})):
 caught=[]
 for seed in range(100):
  try:simulate(3,9,4,seed,**kwargs)
  except AssertionError as error:caught.append({'seed':seed,'reason':str(error)})
 assert caught,mutation
 negative.append({'mutation':mutation,'caught_seed_count':len(caught),'first_failure':caught[0]})
report={'status':'INDEPENDENT_MULTI_WARPGROUP_LIFETIME_MODEL_PASS','positive_schedules':len(positive),'lifetime_assertions':sum(positive),'stages':[3,4],'warp_iters':4,'warpgroups':[1,2,4],'K_tile_counts':[1,2,3,4,5,7,8,9,15,16,17,31,32,33,63,64,65],'negative_controls':negative,'model':['One commit per run, wait1 leaves only newest group pending per warpgroup','Producer overwrite occurs after every warpgroup reaches original penultimate-iteration CTA barrier','Register transform uses alternate dequantized bank; packed prefetch buffers are separate','Final wait0 clears groups before epilogue shared-reduction overwrite/accumulator use'],'limits':['Abstract source-order model, not compiler/SASS/device proof','Assumes admitted A16/WK64 no accumulator postprocessing, no warp specialization/TMA/streamK, identical CTA control flow','Does not prove initial cp.async data readiness; original consumer protocol remains unchanged','No partial element-K tail modeled; renderer must reject nonmultiple BlockK']}
(P/'WAIT1_INDEPENDENT_MULTIWG_MODEL.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'positive_schedules':len(positive),'assertions':sum(positive),'negative':negative}))
