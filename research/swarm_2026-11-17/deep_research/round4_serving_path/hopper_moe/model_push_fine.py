"""Finer model: separate thread producer/poll and row CTA completion."""
import hashlib,json,pathlib,random
P=pathlib.Path(__file__).resolve().parent
W=8;C=132;SIZES=[1,8,2,7,3,8]
def run(seed,negative=''):
 rng=random.Random(seed);mem={};counts=[[0]*C for _ in range(W)];gen=[0]*W;rankstates=[None]*W;events=reads=0
 def begin(r):
  g=gen[r]
  if g==len(SIZES):return None
  assert len(set(counts[r]))==1,('incoherent',r,g)
  phase=counts[r][0];n=SIZES[g];generic=g%2==1
  if generic:
   # Coarsened nativegrid: one activeCTA per vector, remaining counters flip as emptyCTA.
   owners={b:[b] for b in range(4*n)}
   for b in range(4*n,C):counts[r][b]^=1
  else:owners={b:list(range(4*b,4*b+4)) for b in range(n)}
  actors=[]
  for b,vecs in owners.items():
   groups=[vecs] if generic else [vecs[::2],vecs[1::2]]
   for vs in groups:actors.append(dict(block=b,phase=phase,push=[(d,v,a) for v in vs for d in range(W) for a in range(4)],poll=vs[:],done=False))
  return dict(g=g,n=n,generic=generic,actors=actors,flipped=set())
 for r in range(W):rankstates[r]=begin(r)
 while any(g<len(SIZES) for g in gen):
  r=rng.randrange(W);s=rankstates[r]
  if s is None:continue
  live=[a for a in s['actors'] if not a['done']]
  if not live:
   gen[r]+=1;rankstates[r]=begin(r);continue
  a=rng.choice(live);ph=0 if negative=='single_phase' else a['phase'];g=s['g'];b=a['block']
  if a['push']:
   i=rng.randrange(len(a['push']));d,v,j=a['push'].pop(i);key=(d,ph,r,v,j)
   assert mem.get(key,0)==0,('overwrite',seed,g,r,key)
   mem[key]=(g+1)*100000+r*1000+v*4+j+1
  elif a['poll']:
   v=rng.choice(a['poll']);keys=[(r,ph,src,v,j)for src in range(W)for j in range(4)];values=[mem.get(k,0)for k in keys]
   if (any(values) if negative=='partial' else all(values)):
    assert values==[(g+1)*100000+src*1000+v*4+j+1 for src in range(W)for j in range(4)],('stale/partial',seed,r,g)
    for k in keys:mem[k]=0
    a['poll'].remove(v);reads+=32
  else:
   a['done']=True
   if all(x['done'] for x in s['actors'] if x['block']==b):
    assert b not in s['flipped'];s['flipped'].add(b);counts[r][b]^=1
    if not s['generic'] and b==0 and negative!='omit_bumper':
     for j in range(s['n'],C):counts[r][j]^=1
  events+=1;assert events<10000000,'progress bound'
 assert not any(mem.values())
 return events,reads
positives=[run(i) for i in range(20)];neg={}
for mode in ['single_phase','partial','omit_bumper']:
 caught=0;example=None
 for seed in range(20):
  try:run(seed,mode)
  except AssertionError as e:caught+=1;example=example or str(e)
 assert caught>0
 neg[mode]=dict(caught=caught,attempted=20,example=example)
result=dict(status='FINITE_THREAD_ROW_SCHEDULE_MODEL_PASS_NOT_CUDA_PROOF',model_changes=['Each thread publishes its own two packs before polling; other threads may still be producing','Each row flips only after both local thread actors complete','Row0 bumper can advance unused counters while other rowCTAs still run','Generic and fused ownership alternates across ranks with independent progression'],schedules=20,events=sum(x[0] for x in positives),word_reads=sum(x[1] for x in positives),negatives=neg,source_sha256=hashlib.sha256((P/'attention_ar_norm.cuh').read_bytes()).hexdigest(),limitations=['Finite compressed-vector model; full address enumeration remains in earlier receipt','No CUDA system-scope visibility or scheduler guarantee','No speedup evidence'])
(P/'ATTENTION_AR_FINE_PROTOCOL_MODEL.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
