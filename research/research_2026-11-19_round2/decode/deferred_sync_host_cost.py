"""Host bookkeeping lower bound; no Torch, CUDA stream API, GPU or transport.

Executes the exact research helper with its CUDA stream query replaced by a
constant. The native producer is an identity function. Added CPU cost cannot
be interpreted as net NCCL savings, and excludes real CUDA stream API overhead.
"""
import gc,hashlib,importlib.util,json,platform,statistics,sys,time,types
from pathlib import Path
here=Path(__file__).resolve().parent
fake=types.ModuleType('torch');fake.distributed=types.ModuleType('torch.distributed')
sys.modules['torch']=fake;sys.modules['torch.distributed']=fake.distributed
spec=importlib.util.spec_from_file_location('runtime',here/'deferred_sync.py')
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
runtime._stream_state=lambda:(11,False)
class Group:
 world_size=8;rank_in_group=0;ranks=list(range(8));pymscclpp_comm=None
 device_group=object();cpu_group=object()
 def all_gather_into_tensor(self,*args):pass
 def _all_gather_into_tensor(self,*args):pass
 def broadcast(self,*args):pass
class Sync:
 _sites=frozenset(range(1,18))
 def __init__(self,group):self._tp_group=group
 def sync(self,*args):pass
class Processor:
 def deferred_argmax(self,logits):return logits
group=Group();sync=Sync(group);processor=Processor();value=object()
owner=runtime.DeferredArgmaxAgreement(processor,sync,group,Processor,Group,lambda:group)
def native():
 result=processor.deferred_argmax(value)
 return result
def candidate():
 result,proof=owner.produce(processor,sync,value)
 owner.consume(proof,result,sync)
 return result
def control():
 result=processor.deferred_argmax(value)
 return result
def bench(func,n):
 before=time.perf_counter_ns()
 for _ in range(n):assert func() is value
 return (time.perf_counter_ns()-before)/n
for func in (native,candidate,control):bench(func,3000)
gc.disable();rounds=[]
try:
 for i in range(31):
  a1=bench(native,4000);b1=bench(candidate,4000);b2=bench(candidate,4000);a2=bench(native,4000);c=bench(control,4000)
  rounds.append(dict(native_ns=(a1+a2)/2,candidate_ns=(b1+b2)/2,control_ns=c,added_ns=(b1+b2-a1-a2)/2))
finally:gc.enable()
result=dict(scope=__doc__,python=platform.python_version(),platform=platform.platform(),
 source_sha256=hashlib.sha256((here/'deferred_sync.py').read_bytes()).hexdigest(),rounds=rounds,
 median_native_ns=statistics.median(r['native_ns'] for r in rounds),
 median_candidate_ns=statistics.median(r['candidate_ns'] for r in rounds),
 median_control_ns=statistics.median(r['control_ns'] for r in rounds),
 median_added_ns=statistics.median(r['added_ns'] for r in rounds),
 no_gpu=True,no_nccl=True,stream_query_cost_excluded=True,performance_gate='unresolved: actual saved collective latency must exceed all added host cost on serving critical path')
(here/'deferred_sync_host_cost.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rounds'},indent=2))
