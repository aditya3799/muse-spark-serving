from pathlib import Path
from types import SimpleNamespace as NS,ModuleType
import importlib.util,json,sys
import torch
P=Path('/mnt/d/OpenKernelChallenge/serving/swarm_2026-11-16/projection_communication')
spec=importlib.util.spec_from_file_location('validation',P/'validate_row_projection.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
H=P/'candidate_small_multicast/python/sglang/srt/layers/communication/muse_spark_gemm_ag.py'
class Comm:
 def __init__(self,world=8):
  self.disabled=False;self.has_multicast=True;self.world_size=world;self.rank=0;self.max_push_size=128*1024;self.max_pull_size=16*1024*1024;self.config=NS(num_push_blocks=132)
  self._capture_started=False;self._graph_mode_allowed=False;self._graph_counter=0;self._graph_inputs=[];self.device='cuda:0';self.group='cpu';self.initializations=0
 def _init_workspace(self):self.initializations+=1
sync=[];barriers=[]
ns=dict(torch=NS(cuda=NS(synchronize=lambda device:sync.append(device),is_current_stream_capturing=lambda:False)),dist=NS(barrier=lambda group:barriers.append(group)),_FORCE_PUSH_SIZE_KB=None,_ALIGN_BYTES=1024,_ceil_align=lambda x,a:(x+a-1)//a*a)
reserve=v.extract(v.ROOT/'sglang/python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py',{'reserve_push_capacity'},ns,'CustomAllReduceV2')['reserve_push_capacity']
Comm.reserve_push_capacity=reserve
fake=ModuleType('sglang.srt.distributed.device_communicators.custom_all_reduce_v2');fake.CustomAllReduceV2=Comm
sys.modules[fake.__name__]=fake
state={'cc':(9,0)}
fake_torch=NS(bfloat16='bf16',cuda=NS(get_device_capability=lambda device:state['cc']))
helpers=v.extract(H,{'reserve_projection_workspace','small_projection_supported'},{'torch':fake_torch,'MAX_TOKENS':512})
c=Comm()
for layer in range(64):helpers['reserve_projection_workspace'](c,8192)
assert c.initializations==1 and c.max_push_size==8*1024*1024
assert len(sync)==len(barriers)==1
# Repeated no-op reservations remain legal after capture; late growth fails.
c._capture_started=True
helpers['reserve_projection_workspace'](c,8192)
try:c.reserve_push_capacity(16*1024*1024)
except RuntimeError:pass
else:raise AssertionError('late reservation must fail')
# Forced explicit capacity remains authoritative and predicate falls back.
forced=Comm();ns['_FORCE_PUSH_SIZE_KB']=128
helpers['reserve_projection_workspace'](forced,8192)
assert forced.initializations==0 and forced.max_push_size==128*1024
ns['_FORCE_PUSH_SIZE_KB']=None
class T:
 def __init__(self,shape,dtype='bf16',device='cuda:0',cuda=True,contiguous=True):self.shape,self.dtype,self.device,self.is_cuda,self.ndim,self.contig=shape,dtype,device,cuda,len(shape),contiguous
 def is_contiguous(self):return self.contig
checks=[]
for world in (1,2,3,4,8,16):
 comm=Comm(world);comm.max_push_size=8*1024*1024;group=NS(world_size=world,ca_comm=comm)
 for m in (0,1,8,16,32,64,128,256,511,512,1024):
  actual=helpers['small_projection_supported'](group,T((m,8192)),T((4096,8192)))
  expected=world in (2,4,8) and 1<=m<512
  assert actual==expected,(world,m,actual,expected)
  checks.append(dict(world=world,m=m,supported=actual))
for bad_attr,bad_value in (('disabled',True),('has_multicast',False),('config',NS(num_push_blocks=1)),('max_push_size',64)):
 comm=Comm();comm.max_push_size=8*1024*1024;setattr(comm,bad_attr,bad_value)
 assert not helpers['small_projection_supported'](NS(world_size=8,ca_comm=comm),T((32,8192)),T((4096,8192)))
for cc in ((8,6),(10,3)):
 state['cc']=cc
 assert not helpers['small_projection_supported'](NS(world_size=8,ca_comm=c),T((32,8192)),T((4096,8192)))
state['cc']=(9,0)
assert not helpers['small_projection_supported'](NS(world_size=8,ca_comm=forced),T((32,8192)),T((8192,8192)))
# Exact wrapper invokes existing gather once with expected BF16 complete-column slice.
c.rank=3
x=torch.randn(17,128,dtype=torch.bfloat16);w=torch.randn(256,128,dtype=torch.bfloat16)
called=[]
def gather(comm,local,out):
 expected=torch.mm(x.float(),w[96:128].float().t()).to(x.dtype)
 torch.testing.assert_close(local,expected,rtol=0,atol=0)
 assert out.shape==(17,256) and out.dtype==x.dtype
 called.append(True);out.zero_();return out
math=NS(mm=lambda a,b,out_dtype:torch.mm(a.float(),b.float()),float32=torch.float32,empty=torch.empty)
run=v.extract(H,{'small_projection_local','small_projection'},{'torch':math,'all_gather':gather})['small_projection']
run(NS(ca_comm=c),x,w)
assert called==[True]
report=dict(passed=True,shape_cases=checks,workspace_calls=64,workspace_reallocations=c.initializations,workspace_bytes_per_rank=2*8*c.max_push_size+c.max_pull_size,additional_push_workspace_bytes=2*8*(8*1024*1024-128*1024),late_capture_growth_rejected=True,forced_capacity_preserved=True,wrapper_gather_calls=1,scope='CPU exact-source method tests with fake communicator/tensor metadata; no network or CUDA memory measurement.')
(P/'small_cpu_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='shape_cases'},indent=2))
