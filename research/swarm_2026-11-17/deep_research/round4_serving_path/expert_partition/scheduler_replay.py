"""CPU-only causal scheduler replay. Service curves and requests are SYNTHETIC.
No import of SGLang runtime, torch, CUDA, or network code. Not a performance prediction.
"""
import ast,copy,hashlib,json,math,pathlib,random,types
from dataclasses import dataclass,field
from typing import Optional
P=pathlib.Path(__file__).parent;ROOT=P.parents[3]
SCHED=ROOT/'sglang/python/sglang/srt/managers/scheduler.py'

def extract_interval():
 tree=ast.parse(SCHED.read_text(encoding='utf-8'))
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Scheduler')
 wanted=('_should_defer_prefill','_arm_prefill_decode_interval')
 funcs=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
 assert len(funcs)==2
 ns={'Optional':Optional,'ScheduleBatch':object}
 exec(compile(ast.Module(body=funcs,type_ignores=[]),str(SCHED),'exec'),ns)
 return ns
INTERVAL=extract_interval()
class Batch:
 def __init__(self,n,extend=False,sync_extend=None):
  self.n=n;self.forward_mode=types.SimpleNamespace(is_extend=lambda:extend)
  self.is_extend_in_batch=extend if sync_extend is None else sync_extend
 def batch_size(self):return self.n
class FixedInterval:
 def __init__(self,count,min_bs=0,dp=False):
  self.prefill_decode_interval=count;self.prefill_decode_interval_min_bs=min_bs
  self._prefill_decode_interval_remaining=0;self.require_mlp_sync=dp
 def defer(self,n):return INTERVAL['_should_defer_prefill'](self,Batch(n))
 def arm(self,extend):INTERVAL['_arm_prefill_decode_interval'](self,Batch(1,extend))

def control_tests():
 checks=0
 for count in range(5):
  for minbs in (0,1,8):
   for n in (0,1,7,8,32):
    p=FixedInterval(count,minbs);p.arm(True)
    got=[p.defer(n) for _ in range(7)]
    expected=[True]*count+[False]*(7-count) if n>=minbs else [False]*7
    assert got==expected,(count,minbs,n,got,expected);checks+=1
    p.arm(False);assert not p.defer(n);checks+=1
 for dp in (False,True):
  for extend in (False,True):
   for synced in (False,True):
    p=FixedInterval(2,dp=dp)
    INTERVAL['_arm_prefill_decode_interval'](p,Batch(1,extend,synced))
    assert p._prefill_decode_interval_remaining==(2 if (synced if dp else extend) else 0);checks+=1
 return checks

def quantile(v,q):
 if not v:return None
 s=sorted(v);return s[max(0,math.ceil(q*len(s))-1)]
@dataclass
class Req:
 rid:int
 arrival:float
 prompt:int
 output:int
 accepts:list
 remaining_prompt:int=0
 done_tokens:int=0
 prefill_started:float=None
 first:float=None
 finish:float=None
 last_commit:float=None
 next_accept:int=0
 gaps:list=field(default_factory=list)
 token_times:list=field(default_factory=list)
 def __post_init__(self):self.remaining_prompt=self.prompt

def make_requests(seed,n,arrival_mode,shape):
 rng=random.Random(seed);out=[]
 for i in range(n):
  arrival=0.0 if arrival_mode=='burst' else i*3.0+rng.uniform(0,1)
  if shape=='heterogeneous':prompt=rng.choice((256,1024,4096,32768,65536));output=rng.choice((32,64,128))
  elif shape=='long_prefill':prompt=rng.choice((32768,65536));output=64
  elif shape=='decode_heavy':prompt=256;output=256
  else:raise ValueError(shape)
  out.append(Req(i,arrival,prompt,output,[rng.choice((1,2,4,7,8)) for _ in range(output)]))
 return out

def run(requests,policy,curve,chunk=16384,max_running=16,age_budget=12.0,prefill_age_limit=48.0):
 reqs=copy.deepcopy(requests);future=sorted(reqs,key=lambda r:(r.arrival,r.rid));waiting=[];running=[];chunked=None
 fixed=FixedInterval(int(policy.removeprefix('interval')) if policy.startswith('interval') else 0) if hasattr(str,'removeprefix') else FixedInterval(int(policy[len('interval'):]) if policy.startswith('interval') else 0)
 now=0.;trace=[];prefill_tokens=verify_rows=decode_req_rounds=0;prefill_calls=decode_calls=empty_defers=0;idle=0.;iterations=0
 def commit(r,count,t):
  actual=min(count,r.output-r.done_tokens)
  if r.first is None:r.first=t
  if r.last_commit is not None:r.gaps.append(t-r.last_commit)
  r.last_commit=t;r.token_times.extend([t]*actual);r.done_tokens+=actual
  if r.done_tokens==r.output:r.finish=t
 while any(r.finish is None for r in reqs):
  iterations+=1;assert iterations<100000,'liveness failure'
  while future and future[0].arrival<=now+1e-10:waiting.append(future.pop(0))
  running=[r for r in running if r.finish is None]
  has_prefill=chunked is not None or bool(waiting and len(running)<max_running)
  defer=False
  if policy.startswith('interval'):defer=fixed.defer(len(running))
  elif policy=='age_guard' and running and has_prefill:
   oldest_gap=max(now-r.last_commit for r in running)
   # Generic deadline guard with a prefill-wait escape; no request IDs/benchmark lengths.
   oldest_prefill=chunked if chunked is not None else waiting[0]
   wait_age=now-(oldest_prefill.prefill_started if oldest_prefill.prefill_started is not None else oldest_prefill.arrival)
   defer=oldest_gap>=age_budget and wait_age<prefill_age_limit
  elif policy not in ('prefill_first','age_guard'):assert policy.startswith('interval')
  if has_prefill and not defer:
   if chunked is None:
    chunked=waiting.pop(0);chunked.prefill_started=now
   r=chunked;tokens=min(chunk,r.remaining_prompt)
   duration=curve['prefill_launch_ms']+tokens*curve['prefill_ms_per_token']
   start=now;now+=duration;r.remaining_prompt-=tokens;prefill_tokens+=tokens;prefill_calls+=1
   trace.append({'kind':'prefill','start':start,'end':now,'request':r.rid,'tokens':tokens,'running':len(running),'queued':len(waiting)})
   fixed.arm(True)
   if r.remaining_prompt==0:
    commit(r,1,now);running.append(r);chunked=None
  elif running:
   start=now;n=len(running);duration=curve['decode_launch_ms']+n*curve['decode_ms_per_request']
   now+=duration;decode_calls+=1;decode_req_rounds+=n;verify_rows+=8*n
   ids=[]
   for r in running:
    a=r.accepts[r.next_accept];r.next_accept+=1;commit(r,a,now);ids.append(r.rid)
   trace.append({'kind':'decode','start':start,'end':now,'requests':ids,'running':n,'verify_rows':8*n,'queued':len(waiting)})
   fixed.arm(False)
  elif has_prefill and defer:
   empty_defers+=1 # actual source permits bounded empty scheduler turns when min_bs=0
  elif future:
   jump=future[0].arrival-now;assert jump>=0;idle+=jump;now=future[0].arrival
  else:raise AssertionError('deadlock')
 assert prefill_tokens==sum(r.prompt for r in reqs)
 assert sum(r.done_tokens for r in reqs)==sum(r.output for r in reqs)
 expected_rounds=0
 for r in reqs:
  progress=1;i=0
  while progress<r.output:progress+=min(r.accepts[i],r.output-progress);i+=1
  expected_rounds+=i;assert r.next_accept==i
 assert decode_req_rounds==expected_rounds and verify_rows==8*expected_rounds
 gaps=[g for r in reqs for g in r.gaps];ttft=[r.first-r.arrival for r in reqs];e2e=[r.finish-r.arrival for r in reqs]
 summary={'policy':policy,'makespan_ms':now,'idle_ms':idle,'ttft_p90_ms':quantile(ttft,.9),'e2e_p90_ms':quantile(e2e,.9),'commit_gap_p90_ms':quantile(gaps,.9),'commit_gap_p99_ms':quantile(gaps,.99),'max_commit_gap_ms':max(gaps) if gaps else 0,'mean_completion_ms':sum(e2e)/len(e2e),'prefill_calls':prefill_calls,'decode_calls':decode_calls,'prefill_tokens':prefill_tokens,'decode_request_rounds':decode_req_rounds,'verify_rows':verify_rows,'completed_tokens':sum(r.done_tokens for r in reqs),'empty_scheduler_deferrals':empty_defers,'request_completion':[{'rid':r.rid,'ttft_ms':r.first-r.arrival,'e2e_ms':r.finish-r.arrival,'rounds':r.next_accept} for r in reqs]}
 return summary,trace

def main():
 checks=control_tests();rows=[];traces={}
 curves={'additive_no_launch':{'prefill_launch_ms':0,'prefill_ms_per_token':.001,'decode_launch_ms':0,'decode_ms_per_request':.2},'batch_economy':{'prefill_launch_ms':.4,'prefill_ms_per_token':.001,'decode_launch_ms':.8,'decode_ms_per_request':.04},'prefill_dominant':{'prefill_launch_ms':.4,'prefill_ms_per_token':.004,'decode_launch_ms':.8,'decode_ms_per_request':.04}}
 for seed in (11,29,73):
  for shape in ('heterogeneous','long_prefill','decode_heavy'):
   for arrivals in ('burst','staggered'):
    requests=make_requests(seed,24,arrivals,shape)
    for name,curve in curves.items():
     results=[]
     for policy in ('prefill_first','interval1','interval2','interval4','age_guard'):
      result,trace=run(requests,policy,curve);results.append(result)
      if seed==11 and shape=='long_prefill' and arrivals=='burst' and name=='batch_economy':traces[policy]=trace
     baseline=results[0]
     for result in results:
      result['makespan_ratio_to_prefill_first']=result['makespan_ms']/baseline['makespan_ms']
      assert result['prefill_tokens']==baseline['prefill_tokens'] and result['verify_rows']==baseline['verify_rows']
      if name=='additive_no_launch' and arrivals=='burst':assert abs(result['makespan_ms']-baseline['makespan_ms'])<1e-7
     rows.append({'seed':seed,'shape':shape,'arrivals':arrivals,'request_count':24,'max_running':16,'chunk_tokens':16384,'curve':name,'results':results})
 report={'status':'SYNTHETIC_CAUSAL_REPLAY_NOT_SERVING_BENCHMARK','source_sha256':hashlib.sha256(SCHED.read_bytes()).hexdigest(),'actual_extracted_interval_method_checks':checks,'curves_ms':curves,'policy_parameters':{'age_guard_ms':12,'prefill_age_escape_ms':48},'cases':len(rows),'policy_runs':len(rows)*5,'rows':rows,'limits':['No actual request trace or H200 service curve used.','No mixed-step simulation: DFlash mixed extend produces one token, not a full speculative round.','Serial execution model omits overlap and host/KV/communication stalls.','Only one prefill chunk/request per iteration; real scheduler may co-batch short requests.','No numerical model execution; fixed synthetic accepted-prefix streams are identical but real scheduler re-batching requires numerical requalification.','Ages are completed GPU work timestamps in this replay; host enqueue clocks alone cannot supply them in overlap scheduling.','No GPU memory/KV reservation model; bounded max_running is not proof of real cache liveness.']}
 (P/'SCHEDULER_CAUSAL_REPLAY.json').write_text(json.dumps(report,indent=2))
 (P/'SCHEDULER_CAUSAL_TRACE.json').write_text(json.dumps(traces,indent=2))
 print(json.dumps({'checks':checks,'cases':len(rows),'policy_runs':len(rows)*5,'conservation':'all prefill tokens, verify rows, accepted output counts preserved','example':[{'policy':r['policy'],'makespan_ms':r['makespan_ms'],'ttft_p90_ms':r['ttft_p90_ms'],'max_commit_gap_ms':r['max_commit_gap_ms']} for r in rows[4]['results']]},indent=2))
if __name__=='__main__':main()

