"""Real checkpoint full-K/output-N shard qualification and coupled ideal-transport bound.

All eight serial shards are checked against every native FC output before each
timed pair. Only rank0 computes in the one-rank timed arm; other rank outputs
are live-input recomputations outside timing. A local concatenation is charged.
This is explicitly NOT NCCL or a serving speedup measurement.
"""
import argparse
import fcntl
import hashlib
import inspect
import json
import statistics
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'research_2026-11-18_lt_audit'))
sys.path.insert(0,str(ROOT/'research_2026-11-18_rl'))
import real_coupled as real
import coupled_full as coupled
from native_abi import load_lib,describe
from runtime_projection import native_workspace
from run_pair import telemetry

class OutputShard:
 def __init__(self,w,m,seed,frozen=None):
  self.w=w;self.m=m;self.width=w.shape[0]//8;self.lib=load_lib();self.index=None
  self.reason='unqualified';self.full_bytes=0;self.stream=torch.cuda.current_stream().cuda_stream
  self.budget=native_workspace()[1]
  self.q=self.lib.make_sweep(m,self.width,w.shape[0],w.shape[1],self.width,self.budget,self.budget,0)
  self.info=dict(m=m,full_n=w.shape[0],k=w.shape[1],world=8,native_workspace=native_workspace(),tried=[])
  self.validation=[];self.peer=[]
  if not self.q:self.reason='no vendor plan';return
  gen=torch.Generator(device='cuda').manual_seed(seed)
  examples=[torch.empty((m,w.shape[1]),device='cuda',dtype=w.dtype).normal_(generator=gen) for _ in range(2)]
  # Signed, large-dynamic-range activations test every native output channel.
  examples[1][:,::4]*=32;examples[1][:,1::4]*=.03125
  refs=[F.linear(x,w) for x in examples]
  for i in range(self.lib.sweep_count(self.q)):
   desc=describe(self.lib,self.q,i)
   if desc['source']!=0 or desc['workspace'][0]!=self.budget:continue
   if frozen is not None and any(desc[k]!=frozen[k] for k in ('id','tile','split','reduction','swizzle','custom','stages','inner','cluster')):continue
   self.temp=torch.empty(max(desc['workspace'][1:]),device='cuda',dtype=torch.uint8)
   checks=[]
   for x,ref in zip(examples,refs):
    full=self.invoke(i,True,x,w)
    counts=[int((full.view(torch.int16)!=ref.view(torch.int16)).sum())]
    for rank in range(8):
     out=self.invoke(i,False,x,w[rank*self.width:(rank+1)*self.width])
     counts.append(int((out.view(torch.int16)!=ref[:,rank*self.width:(rank+1)*self.width].view(torch.int16)).sum()))
    checks.append(counts)
   self.info['tried'].append(dict(config=desc,mismatches=checks))
   if not any(any(c) for c in checks):
    self.index=i;self.config=desc;self.full_bytes=desc['workspace'][2]
    self.reason='qualified finite local tests; no mathematical cross-dimension guarantee'
    self.info['selected']=desc;break
  if self.index is None:self.reason='no matching full-parent plan'
 def invoke(self,index,full,x,w):
  n=self.w.shape[0] if full else self.width
  out=torch.empty((self.m,n),device=x.device,dtype=x.dtype)
  status=self.lib.sweep_call(self.q,index,int(full),w.data_ptr(),x.data_ptr(),out.data_ptr(),self.temp.data_ptr(),self.temp.numel(),torch.cuda.current_stream().cuda_stream)
  if status:raise RuntimeError(('vendor call',status))
  return out
 def all_shards(self,x):
  return [self.invoke(self.index,False,x,self.w[r*self.width:(r+1)*self.width]) for r in range(8)]
 def check(self,x):
  self.initial_bad=0 if self.index is not None else None
 def prepare_peers(self,x):
  if self.index is None:return
  self.peer=self.all_shards(x)
  full=F.linear(x,self.w)
  mismatches=[int((p.view(torch.int16)!=full[:,r*self.width:(r+1)*self.width].view(torch.int16)).sum()) for r,p in enumerate(self.peer)]
  self.validation.append(mismatches)
  if any(mismatches):raise AssertionError(('live all-rank FC mismatch',mismatches))
 def execute(self,x,w):
  if self.index is None:return F.linear(x,w)
  if (tuple(x.shape)!=(self.m,self.w.shape[1]) or not x.is_contiguous() or x.dtype!=self.w.dtype
      or x.device!=self.w.device or w.data_ptr()!=self.w.data_ptr() or tuple(w.shape)!=tuple(self.w.shape)
      or w.stride()!=self.w.stride() or torch.cuda.current_stream().cuda_stream!=self.stream
      or torch.cuda.current_device()!=self.w.device.index):
   return F.linear(x,w)
  if not self.peer:self.prepare_peers(x)
  own=self.invoke(self.index,False,x,w[:self.width])
  return torch.cat([own,*self.peer[1:]],dim=1)
 def close(self):
  if self.q:self.lib.drop_sweep(self.q);self.q=None

 def trace(self,directory,seed):
  gen=torch.Generator(device='cuda').manual_seed(seed)
  x=torch.randn(self.m,self.w.shape[1],device='cuda',dtype=self.w.dtype,generator=gen)
  records={}
  calls={'native':lambda:F.linear(x,self.w),
   'full_parent_plan':lambda:self.invoke(self.index,True,x,self.w),
   'output_N_shard':lambda:self.invoke(self.index,False,x,self.w[:self.width])}
  for name,fn in calls.items():
   fn();torch.cuda.synchronize()
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    fn();torch.cuda.synchronize()
   path=directory/f'trace_M{self.m}_{name}.json';prof.export_chrome_trace(str(path))
   events=json.loads(path.read_text())['traceEvents']
   records[name]=[dict(name=e['name'],args=e.get('args',{})) for e in events if e.get('cat')=='kernel']
  return records

def case_fn(k_norm):
 source=inspect.getsource(coupled.pinned.run_case)
 # Reuse the already audited real-weight coupled transformation, then insert
 # a live-input virtual-peer refresh before each timed reciprocal group.
 fn=coupled.patched(checkpoint_k_norm=k_norm,track_changes=True)
 # The generated function does not retain inspectable source. Patch its
 # generator source once, preserving the checked replacement contracts.
 generator=inspect.getsource(coupled.patched)
 needle='    scope={**ENV,'
 addition='''    source=source.replace("    for name in ARMS:\\n        launch(name)", "    producer[\\\"original\\\"][0].replay()\\n    fcplan.prepare_peers(producer[\\\"original\\\"][1][1])\\n    for name in ARMS:\\n        launch(name)")
    source=source.replace("        order_log.append(order)", "        producer[\\\"original\\\"][0].replay()\\n        fcplan.prepare_peers(producer[\\\"original\\\"][1][1])\\n        order_log.append(order)")
'''
 assert generator.count(needle)==1
 generator=generator.replace(needle,addition+needle)
 ns=dict(coupled.__dict__);exec(compile(generator,'output_shard_generator.py','exec'),ns)
 return ns['patched'](checkpoint_k_norm=k_norm,track_changes=True)

def main(a,directory):
 # New receipts point to the one hash-verified shared recovery. Historical
 # receipt paths and their now-absent assets remain untouched.
 receipts=directory/'asset_receipts';receipts.mkdir()
 recovered=ROOT/'research_2026-11-19/prefill'
 for name in ('public_fc_receipt.json','public_aux_receipt.json'):
  (receipts/name).write_bytes((recovered/('recovered_'+name)).read_bytes())
 real.SOURCE=receipts
 (fc,kv,norm,k_norm),weight_info=real.checkpoint('cuda')
 rope=real.configured_rope('cuda',weight_info['rope_theta']);rms=coupled.rms_linux();run=case_fn(k_norm)
 doc=dict(agent='decode_research',candidate='real_checkpoint_output_N_shard_ideal_transport_bound',
  scope='Actual DFlash weights, full-K dot products at exact M, full output restored before native RMS and five original KV operations. One timed rank plus live virtual-peer outputs and charged local concatenate; excludes NCCL and target/draft blocks. All8serial shards checked each replay.',
  baseline_sha256=real.BASELINE,seed=a.seed,weights=weight_info,correct=True,active=True,complete=False,
  bench=dict(instances=1,shapes={}),details={},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 def save():(directory/'result.json').write_text(json.dumps(doc,indent=2))
 save()
 frozen=json.loads(Path(a.frozen).read_text()) if a.frozen else None
 for b in a.batches:
  config=frozen['details'][f'B{b}']['plan']['selected'] if frozen else None
  plan=OutputShard(fc,b*8,a.seed+b,config)
  name=f'B{b}';doc['details'][name]=dict(plan=plan.info)
  if plan.index is None:doc['active']=False;save();plan.close();continue
  try:
   for risk in a.risks:
    start=len(plan.validation)
    row=run(b,a.seed,fc,kv,norm,rope,rms,a.rounds,plan,risk)
    row['all_rank_live_mismatches']=plan.validation[start:]
    key=name if len(a.risks)==1 else name+'_'+risk
    doc['details'][name]['coupled_'+risk]=row
    doc['bench']['shapes'][key]=dict(weight=1,times=row['wall_us'])
    doc['correct']&=not any(row['wrong'].values()) and not any(any(s.values()) for s in row['wrong_stages'].values())
    print(key,plan.config,'correct',doc['correct'],{k:statistics.median(v) for k,v in row['wall_us'].items()},flush=True)
    save()
   if a.trace:doc['details'][name]['kernel_trace']=plan.trace(directory,a.seed+b*7);save()
  finally:plan.close()
  torch.cuda.empty_cache()
 doc['complete']=True;save()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--batches',nargs='+',type=int,default=[1,4,16,32]);p.add_argument('--rounds',type=int,default=8)
 p.add_argument('--risks',nargs='+',choices=['partial','full'],default=['partial']);p.add_argument('--frozen');p.add_argument('--trace',action='store_true');a=p.parse_args()
 assert a.rounds%2==0
 directory=HERE/f'output_shard_{a.seed}';directory.mkdir(exist_ok=False)
 with (ROOT/'research_2026-11-18_rl/.gpu.lock').open('a+') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX)
  with telemetry(directory),torch.inference_mode():main(a,directory)
