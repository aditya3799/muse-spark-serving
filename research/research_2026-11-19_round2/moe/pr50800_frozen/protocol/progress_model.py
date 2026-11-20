"""Abstract progress model of the donor's monotone, resident-claim, FIFO tile queue.
This is not CUDA synchronization or multicast verification.
"""
import random,json,pathlib,hashlib
P=pathlib.Path(__file__).resolve().parent
source=P/'sources/nov10_sweep/pr50800/nvfp4_w4a16_sm90.cuh'
s=source.read_text();assert 'if (lane == 0) t0 = atomicAdd(counter, 1);' in s and 'while (ld_acquire(&cnt1[jt]) < S.RB1)' in s

def run(jobs,workers,seed,*,prefilled=False,reverse=False):
 rng=random.Random(seed);queues=[[] for _ in range(workers)];done=set();cursor=0;steps=0;max_pending=0
 if prefilled: # Counterexample: nonscheduled CTA owns a predecessor.
  blocked=jobs[0];cursor=1
 while True:
  if len(done)==len(jobs):return True,steps,max_pending
  actions=[]
  for w,q in enumerate(queues):
   if cursor<len(jobs) and len(q)<8:actions.append(('claim',w))
   if q:
    item=q[-1] if reverse else q[0]
    if all(dep in done for dep in item[1]):actions.append(('finish',w))
  if not actions:return False,steps,max_pending
  action,w=rng.choice(actions);q=queues[w]
  if action=='claim':q.append(jobs[cursor]);cursor+=1
  else:
   item=q.pop(-1 if reverse else 0);done.add(item[0])
  steps+=1;max_pending=max(max_pending,sum(map(len,queues)))
  assert steps<=2*len(jobs)+workers

def jobs_for(groups,upblocks,downblocks):
 ups=[(('up',g,u),()) for g in range(groups) for u in range(upblocks)]
 downs=[(('down',g,d),tuple(('up',g,u) for u in range(upblocks))) for g in range(groups) for d in range(downblocks)]
 return ups+downs
results=[]
for seed in range(32):
 for workers in (1,2,3,8,16):
  for groups,up,down in ((1,1,1),(7,3,2),(32,64,16),(0,64,16)):
   jobs=jobs_for(groups,up,down);ok,steps,pending=run(jobs,workers,seed)
   assert ok;results.append(dict(seed=seed,workers=workers,groups=groups,upblocks=up,downblocks=down,steps=steps,max_pending=pending))
# A dependency-first queue ordering is essential: down then up on one residentworker deadlocks.
bad=jobs_for(1,1,1);bad=[bad[1],bad[0]];bad_ok,_,_=run(bad,1,0);assert not bad_ok
# A static assignment to a never-scheduled CTA defeats resident-only progress.
static_ok,_,_=run(jobs_for(1,1,1),1,0,prefilled=True);assert not static_ok
r=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cases=len(results),all_passed=True,maximum_jobs=32*(64+16),total_actions=sum(x['steps'] for x in results),workers=[1,2,3,8,16],pending_capacity=8,negative_controls=dict(dependency_before_producer_deadlocks=not bad_ok,unresident_static_owner_deadlocks=not static_ok),assumptions=['Only resident workers claim monotonically from the counter.','Every GEMM1 tile is globally before every GEMM2 tile.','Each worker consumes its queue in FIFO order.','A ready producer/copy-out eventually makes progress; active workers are not permanently descheduled.','Publication/counter visibility is correct and all dependencies are enumerated.'],not_proven=['CUDA mbarrier/TMA async-proxy visibility','cluster cross-CTA progress','PDL ordering','compiler instruction lifetime','real routing output correctness','capture/mixed-shape lifetime','GPU performance'],results=results)
(P/'pr50800_progress_model.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k!='results'})
