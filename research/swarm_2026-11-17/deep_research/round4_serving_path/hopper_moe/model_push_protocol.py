"""Adversarial atomic-word push-plane model. Not a CUDA memory-model proof."""
import collections,hashlib,json,pathlib,random,struct
HERE=pathlib.Path(__file__).resolve().parent
W=8;COUNTERS=132

def check_layout():
 checks=0
 for m in range(1,9):
  fused=[(row,tid+p*256,row*1024+tid+p*256) for row in range(m) for tid in range(256) for p in range(4)]
  assert sorted(v for _,_,v in fused)==list(range(m*1024))
  for threads in [32,64,128,256,512,1024]:
   native=[]
   for b in range(COUNTERS):
    for t in range(threads):
     start=(b+COUNTERS*(t//32))*32+t%32
     native.extend(range(start,m*1024,threads*COUNTERS))
   assert sorted(native)==list(range(m*1024));checks+=m*1024
 return checks

def simulate(seed,broken=''):
 rng=random.Random(seed); rounds=12
 # Compressed vector count preserves changing CTA ownership/phase; layout test separately uses H8192.
 sizes=[1,8,2,7,3,8,1,4,8,2,6,1]
 current=[0]*W;cnt=[[0]*COUNTERS for _ in range(W)]
 stage=[None]*W; mem={}; reads=0;events=0
 def init(r):
  g=current[r]
  if g==rounds:return None
  n=sizes[g]; generic=g%3==1
  active=COUNTERS if generic else n
  assert len(set(cnt[r]))==1, ('incoherent counters',r,g)
  return dict(g=g,n=n,active=active,phase=cnt[r][0]&1,push=[(d,s,a) for d in range(W) for s in range(n) for a in range(4)],poll=list(range(n)),flipped=False)
 for r in range(W):stage[r]=init(r)
 idle=0
 while any(g<rounds for g in current):
  r=rng.randrange(W);s=stage[r]
  if s is None:continue
  g=s['g'];ph=0 if broken=='single_phase' else s['phase']; progressed=False
  # Random stores are 32-bit atoms; consumers cannot read a partly arrived 16B vector.
  if s['push']:
   i=rng.randrange(len(s['push']));d,v,a=s['push'].pop(i)
   key=(d,ph,r,v,a)
   if mem.get(key,0)!=0:raise AssertionError(('overwrite',seed,g,r,key,mem[key]))
   mem[key]=(g+1)*100000+r*1000+v*4+a+1
   progressed=True
  elif s['poll']:
   i=rng.randrange(len(s['poll']));v=s['poll'][i]
   values=[mem.get((r,ph,src,v,a),0) for src in range(W) for a in range(4)]
   ready=any(values) if broken=='partial_vector' else all(values)
   if ready:
    expected=[(g+1)*100000+src*1000+v*4+a+1 for src in range(W) for a in range(4)]
    assert values==expected,('mixed/stale/partial',seed,g,r,v,values,expected)
    for src in range(W):
     for a in range(4):mem[(r,ph,src,v,a)]=0
    s['poll'].pop(i);reads+=W*4;progressed=True
  else:
   # Generic flips every counter; fused rows flip owned counters plus bumper range.
   limit=s['active'] if broken=='omit_bumper' else COUNTERS
   for b in range(limit):cnt[r][b]^=1
   current[r]+=1;stage[r]=init(r);progressed=True
  events+=1
  idle=0 if progressed else idle+1
  assert idle<100000, ('deadlock',seed,current)
 assert not any(mem.values())
 return dict(events=events,word_reads=reads)

def main():
 positives=[simulate(s) for s in range(100)]
 negative={}
 for mode in ['single_phase','omit_bumper','partial_vector']:
  caught=0;first=None
  for seed in range(50):
   try:simulate(seed,mode)
   except AssertionError as e:caught+=1;first=first or str(e)[:300]
  negative[mode]=dict(caught=caught,total=50,first=first)
  assert caught>0,mode
 report=dict(status='CPU_PROTOCOL_MODEL_PASS_NOT_CUDA_VALIDATION',layout_checks=check_layout(),positive_schedules=100,events=sum(x['events'] for x in positives),word_reads=sum(x['word_reads'] for x in positives),negative_controls=negative,assumptions=['All rank calls have same shape and generation sequence','Each rank starts next collective only after all prior local CTAs finish','32-bit system writes become visible atomically; actual device ordering not proved','Every local CTA is eventually scheduled; real residency/deadlock requires runtime check','Two phases shared with generic push; all132 counters must remain coherent'],source_sha256=hashlib.sha256((HERE/'attention_ar_norm.cuh').read_bytes()).hexdigest())
 (HERE/'ATTENTION_AR_PROTOCOL_MODEL.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
