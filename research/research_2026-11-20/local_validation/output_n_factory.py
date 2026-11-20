"""Actual-weight packaged runtime/graph/guard test, single-GPU virtual peers.

Virtual peers recompute live native outputs inside each captured graph; this
tests graph storage lifetimes and local fallback control, not distributed NCCL.
"""
import json,sys,hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent
import sys,types,importlib
ROOT=Path('/mnt/d/OpenKernelChallenge/serving')
SOURCE=ROOT/'research_2026-11-19_final/assembled/python/sglang'
for name,p in [('sglang',SOURCE),('sglang.srt',SOURCE/'srt'),('sglang.srt.speculative',SOURCE/'srt/speculative')]:
 m=types.ModuleType(name);m.__path__=[str(p)];sys.modules[name]=m
sys.path.insert(0,str(ROOT/'research_2026-11-19/prefill'))

sys.path.insert(0,str(ROOT/'research_2026-11-18_lt_audit'))
import real_coupled as real
from sglang.srt.speculative.dflash_projection_n import create_projection
from sglang.srt.speculative.dflash_projection_n import adapter,entry
receipts=HERE/'factory_receipts';receipts.mkdir(exist_ok=True)
for name in ('public_fc_receipt.json','public_aux_receipt.json'):
 (receipts/name).write_bytes((ROOT/'research_2026-11-19/prefill'/('recovered_'+name)).read_bytes())
real.SOURCE=receipts

class VirtualGroup:
 world_size=8;rank_in_group=0;cpu_group=None
 def __init__(self,model):self.model=model;self.x=None;self.calls=0;self.slow=False
 def all_gather(self,local,dim):
  self.calls+=1;native=F.linear(self.x,self.model.weight);w=native.shape[1]//8
  if self.slow:torch.cuda._sleep(2000000)
  return torch.cat([local if r==self.rank_in_group else native[:,r*w:(r+1)*w] for r in range(8)],dim=dim)

def publish(outputs,value,group=None):
 for r in range(len(outputs)):outputs[r]=value

@torch.no_grad()
def main():
 # Only startup CPU collectives are virtualized. Vendor calls and graph
 # replay are real; the separate Gloo test covers rank-asymmetric control.
 adapter.dist.all_gather_object=publish;adapter.dist.all_reduce=lambda value,**kwargs:None
 (fc,kv,norm,k_norm),weights=real.checkpoint('cuda');del kv,norm,k_norm
 model=torch.nn.Linear(40960,8192,bias=False,device='meta',dtype=torch.bfloat16)
 model.weight=torch.nn.Parameter(fc,requires_grad=False);group=VirtualGroup(model)
 router=create_projection(model,group,mode='qualified',rows=(8,32,128,256,2**30),measure=False)
 if router is None:raise AssertionError('package prerequisite failed')
 doc=dict(scope=__doc__,weights=weights,correct=True,rows={},peak_memory_bytes=0,
          resource_rejected_huge_row=(2**30,40960) not in router.entries)
 original_local=router._local;dispatch=[]
 def track(e,x):
  y=original_local(e,x);dispatch.append(y is not None);return y
 router._local=track
 gen=torch.Generator(device='cuda').manual_seed(7381951)
 for m in (8,32,128,256):
  key=(m,40960);e=router.entries.get(key)
  assert e is not None,('no packaged plan',m)
  x=torch.randn(key,device='cuda',dtype=torch.bfloat16,generator=gen);group.x=x
  row=dict(metadata=e.metadata,checks=[])
  for rank in range(8):
   group.rank_in_group=rank;y=router(x)
   row['checks'].append(dict(kind='rank',rank=rank,exact=torch.equal(y,F.linear(x,fc)),fast=dispatch[-1]))
  group.rank_in_group=0
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   router(x);router(x)
  torch.cuda.current_stream().wait_stream(stream)
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph,stream=stream):out=router(x)
  for variant in ('normal','integer','scaled'):
   if variant=='integer':x.random_(-3,4,generator=gen)
   else:x.normal_(generator=gen);x.mul_(16 if variant=='scaled' else 1)
   graph.replay();torch.cuda.synchronize()
   row['checks'].append(dict(kind='graph_'+variant,exact=torch.equal(out,F.linear(x,fc))))
  del graph,out
  if m==8:
   pointer=e.weight_pointer;e.weight_pointer+=256
   y=router(x);row['checks'].append(dict(kind='pointer_guard',exact=torch.equal(y,F.linear(x,fc)),fallback=not dispatch[-1]));e.weight_pointer=pointer
   state=e.native_state;e.native_state=(state[0],not state[1],*state[2:])
   y=router(x);row['checks'].append(dict(kind='math_policy_guard',exact=torch.equal(y,F.linear(x,fc)),fallback=not dispatch[-1]));e.native_state=state
   call=e.plan.invoke;e.plan.invoke=lambda *args:15
   y=router(x);row['checks'].append(dict(kind='vendor_status_guard',exact=torch.equal(y,F.linear(x,fc)),fallback=not dispatch[-1]));e.plan.invoke=call
   storage=torch.randn((m,81920),device='cuda',dtype=torch.bfloat16,generator=gen);strided=storage[:,::2];group.x=strided
   y=router(strided);row['checks'].append(dict(kind='stride_guard',exact=torch.equal(y,F.linear(strided,fc)),fallback=not dispatch[-1]));group.x=x
   del storage,strided
  doc['rows'][str(m)]=row
  doc['correct']&=all(c['exact'] and c.get('fast',True) and c.get('fallback',True) for c in row['checks'])
  print('PACKAGE',m,doc['correct'],flush=True)
  del x,y
  torch.cuda.empty_cache()
 doc['scratch_bytes']=sum(t.numel() for t in router.stream_workspaces.values())
 doc['scratch_buffers']=len(router.stream_workspaces)
 try:router.retire()
 except RuntimeError:doc['retire_requires_destroyed_graphs']=True
 else:doc['retire_requires_destroyed_graphs']=False;doc['correct']=False
 router.retire(graphs_destroyed=True)
 # A deliberately slow virtual gather must not publish an otherwise exact
 # plan. Its fake peers compute all columns with the native FC inside timing;
 # this is a gate/control test, explicitly not a transport benchmark.
 group.slow=True
 old_timing=entry._startup_timing
 def timing_with_live_peers(r,value,rounds=12):
  group.x=value
  return old_timing(r,value,rounds)
 entry._startup_timing=timing_with_live_peers
 gated=create_projection(model,group,mode='qualified',rows=(8,),measure=True)
 doc['slow_collective_disabled']=gated is not None and gated.entries.get((8,40960)) is None
 doc['correct']&=doc['slow_collective_disabled'] and doc['resource_rejected_huge_row']
 gated.retire(graphs_destroyed=True)
 doc['peak_memory_bytes']=torch.cuda.max_memory_allocated()
 assert doc['correct'], doc
 doc['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 doc['package_sources']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE/'srt/speculative/dflash_projection_n').glob('*') if p.suffix in ('.py','.cc')}
 (HERE/'output_n_package_probe_v2.json').write_text(json.dumps(doc,indent=2));print(json.dumps({'correct':doc['correct'],'peak':doc['peak_memory_bytes'],'scratch':doc['scratch_bytes']}))

if __name__=='__main__':main()
