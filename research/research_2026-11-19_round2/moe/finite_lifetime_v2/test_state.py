import json,sys
from pathlib import Path
from finite_scale_state import ImmutableWeightsLease,scan_and_certify
assert 'torch' not in sys.modules
class Tensor:
 def __init__(self,p):self.pointer=p;self._version=0;self.dtype='torch.uint8';self.device='cuda:0';self.size=9216
 def data_ptr(self):return self.pointer
 def numel(self):return self.size
class Layer:
 def __init__(self):self.w13=Tensor(1000);self.w2=Tensor(2000)
def setup():
 x=Layer();lease=ImmutableWeightsLease(x);p=scan_and_certify(x,lease,lambda _:True);assert p.matches(x);return x,lease,p
cases=[]
for name,fn in [('version',lambda x:setattr(x.w13,'_version',1)),('replacement',lambda x:setattr(x,'w13',Tensor(1000))),('pointer',lambda x:setattr(x.w2,'pointer',3000)),('size',lambda x:setattr(x.w2,'size',3)),('dtype',lambda x:setattr(x.w2,'dtype','torch.int32')),('device',lambda x:setattr(x.w2,'device','cuda:1'))]:
 x,lease,p=setup();fn(x);assert not p.matches(x)
 try:
  with p.execution(x):raise AssertionError('invalid dispatch allowed')
 except RuntimeError:pass
 cases.append(name)
for name,fn in [('mutation_during_scan',lambda x:setattr(x.w13,'_version',1)),('replacement_during_scan',lambda x:setattr(x,'w13',Tensor(1000)))]:
 x=Layer();lease=ImmutableWeightsLease(x)
 def scan(x):fn(x);return True
 assert scan_and_certify(x,lease,scan) is None;cases.append(name)
for result in (False,None,1,'true'):
 x=Layer();assert scan_and_certify(x,ImmutableWeightsLease(x),lambda _:result) is None;cases.append('unverified_'+repr(result))
x,lease,p=setup();seen=[]
lease.retire(retire_graphs_and_synchronize=lambda:seen.append(not p.matches(x)))
assert seen==[True] and not p.matches(x) and scan_and_certify(x,lease,lambda _:True) is None;cases.append('retire_invalidates_before_sync')
x=Layer();lease=ImmutableWeightsLease(x)
def invalidate_scan(_):lease.retire(retire_graphs_and_synchronize=lambda:None);return True
assert scan_and_certify(x,lease,invalidate_scan) is None;cases.append('retired_during_scan')
x,lease,p=setup();assert not p.matches(Layer());cases.append('wrong_owner')
x=Layer();del x.w13._version;assert scan_and_certify(x,ImmutableWeightsLease(x),lambda _:True) is None;cases.append('missing_version')
class Inference(Tensor):
 @property
 def _version(self):raise RuntimeError('inference tensor')
 @_version.setter
 def _version(self,v):pass
x=Layer();x.w13=Inference(1000);assert scan_and_certify(x,ImmutableWeightsLease(x),lambda _:True) is None;cases.append('inference_version')
x,lease,p=setup()
try:lease.retire(retire_graphs_and_synchronize=None);raise AssertionError('missing sync accepted')
except TypeError:pass
cases.append('sync_callback_required')
x,lease,p=setup();assert ImmutableWeightsLease(x) is lease;cases.append('canonical_live_lease')
with p.execution(x):
 try:lease.retire(retire_graphs_and_synchronize=lambda:None);raise AssertionError('reentrant retirement accepted')
 except RuntimeError:pass
 assert p.matches(x)
assert lease.active_enqueues==0;cases.append('reentrant_retirement_rejected')
lease.retire(retire_graphs_and_synchronize=lambda:None)
newlease=ImmutableWeightsLease(x);assert newlease is lease and not newlease.live and not p.matches(x);cases.append('retired_owner_cannot_be_recertified')
report=dict(complete=True,negative_cases=cases,torch_free_import=True,runtime_enabled=False,untracked_writes_require_owner_contract=True,graph_replay_requires_host_execution_guard=True)
Path(__file__).with_name('cpu_contracts.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
