"""Collective-safe full-K/output-N DFlash FC research adapter.

Explicitly register identical full shapes on every TP rank before graph capture,
using multiple live calibration inputs and the already loaded original weight.
No candidate is activated by importing this file. H200 qualification is pending.
The qualified mode uses finite local parity gates, not a universal proof of
vendor floating-point equivalence. Shadow mode retains the original result on
any live mismatch, for diagnostics; its overhead must never be omitted in timing.
"""
from dataclasses import dataclass,field
import ctypes
import torch
import torch.distributed as dist
from runtime_projection import discover,compatible,native_workspace

def runtime_state():
 return (native_workspace(),torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
         torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled())

def weight_version(w):
 try:return w._version
 except RuntimeError:return None  # inference tensor: static checkpoint only

@dataclass
class Entry:
 plan: object
 native_state: object
 weight_pointer: int
 weight_version: int
 weight_shape: tuple
 weight_stride: tuple
 workspaces: dict=field(default_factory=dict)
 metadata: dict=field(default_factory=dict)

class OutputNRouter:
 def __init__(self,original,group,*,mode='shadow'):
  if not isinstance(original,torch.nn.Linear) or original.bias is not None:
   raise TypeError('requires the original bias-free native Linear')
  if mode not in ('shadow','qualified'):raise ValueError(mode)
  if group.world_size!=8:raise ValueError('this research gate is TP8 only')
  self.original=original;self.group=group;self.mode=mode;self.entries={}
  self.closed=False;self.stream_workspaces={};self.registration_index=0;self.manifest=None

 def prepare_registration(self,shapes):
  try:local=dict(shapes=[tuple(s) for s in shapes],mode=self.mode,error=None)
  except Exception as exc:local=dict(shapes=None,mode=self.mode,error=repr(exc))
  declarations=[None]*self.group.world_size
  dist.all_gather_object(declarations,local,group=self.group.cpu_group)
  accepted=all(d['error'] is None and d['shapes']==declarations[0]['shapes'] and d['mode']==declarations[0]['mode'] for d in declarations)
  self.manifest=declarations[0]['shapes'] if accepted else None
  self.registration_index=0
  return accepted,declarations

 def register(self,examples,*,discover_fn=discover):
  """Collective startup operation, never called automatically from a forward."""
  key=None;w=self.original.weight
  try:
   if self.closed or torch.cuda.is_current_stream_capturing():raise ValueError('registration requires live router outside capture')
   examples=list(examples)
   if len(examples)<2:raise ValueError('two or more calibration inputs required')
   x=examples[0];key=tuple(x.shape)
   if key in self.entries:raise ValueError('shape already registered')
   if self.manifest is None or self.registration_index>=len(self.manifest) or self.manifest[self.registration_index]!=key:raise ValueError('startup manifest mismatch')
   signature=dict(shape=key,weight_shape=tuple(w.shape),dtype=str(x.dtype),weight_dtype=str(w.dtype),
                  world=self.group.world_size,mode=self.mode,index=self.registration_index,error=None)
  except Exception as exc:signature=dict(shape=key,index=self.registration_index,error=repr(exc))
  signatures=[None]*self.group.world_size
  dist.all_gather_object(signatures,signature,group=self.group.cpu_group)
  self.registration_index+=1
  if any(s.get('error') is not None or s!=signatures[0] for s in signatures):
   return dict(globally_enabled=False,reason='inconsistent registration',ranks=signatures)
  plan=None;metadata={}
  try:
   plan,metadata=discover_fn([(v,w) for v in examples],self.group.world_size)
   if plan is not None:
    state=runtime_state();stream=torch.cuda.current_stream().cuda_stream
    identity=(w.data_ptr(),weight_version(w),tuple(w.shape),w.stride())
    workspace_key=(w.device,stream,state[0][1])
    if workspace_key not in self.stream_workspaces:
     self.stream_workspaces[workspace_key]=torch.empty(state[0][1],device=w.device,dtype=torch.uint8)
  except Exception as exc:
   if plan is not None:plan.close();plan=None
   metadata={'reason':'local qualification failed','error':repr(exc)}
  available=torch.tensor(int(plan is not None),dtype=torch.int32)
  dist.all_reduce(available,op=dist.ReduceOp.MIN,group=self.group.cpu_group)
  if not int(available):
   if plan is not None:plan.close()
   self.entries[key]=None
   return dict(metadata,globally_enabled=False)
  entry=Entry(plan,state,*identity,
              {stream:self.stream_workspaces[workspace_key]},dict(metadata,globally_enabled=True))
  # Discovery's smaller per-plan allocation has no captured users yet.
  plan.workspace=None
  self.entries[key]=entry
  return entry.metadata

 def _local(self,entry,x):
  w=self.original.weight;plan=entry.plan
  if (not compatible(x,w,8) or x.device!=plan.device or
      tuple(x.shape)!=(plan.m,plan.k) or w.data_ptr()!=entry.weight_pointer or
      weight_version(w)!=entry.weight_version or tuple(w.shape)!=entry.weight_shape or
      w.stride()!=entry.weight_stride or runtime_state()!=entry.native_state):return None
  stream=torch.cuda.current_stream().cuda_stream
  workspace=entry.workspaces.get(stream)
  if workspace is None:
   # Graph warmup binds a separate buffer on its own stream. A capture that
   # skipped warmup uses the collective-safe native-slice branch below.
   if torch.cuda.is_current_stream_capturing():return None
   workspace_key=(w.device,stream,entry.native_state[0][1])
   if workspace_key not in self.stream_workspaces:
    self.stream_workspaces[workspace_key]=torch.empty(entry.native_state[0][1],device=w.device,dtype=torch.uint8)
   workspace=self.stream_workspaces[workspace_key]
   entry.workspaces[stream]=workspace
  width=plan.n//8;rank=self.group.rank_in_group
  if not 0<=rank<8:raise ValueError('invalid TP rank')
  out=torch.empty((plan.m,width),device=x.device,dtype=x.dtype)
  status=plan.invoke(plan.index,False,w.narrow(0,rank*width,width),x,out,workspace)
  return None if status else out

 def __call__(self,x):
  if self.closed:raise RuntimeError('router retired')
  entry=self.entries.get(tuple(x.shape))
  if entry is None:return self.original(x)
  try:local=self._local(entry,x)
  except (ValueError,TypeError,AttributeError,ctypes.ArgumentError,torch.cuda.OutOfMemoryError):local=None
  # A sticky CUDA/context error is not recoverable; let the worker abort rather
  # than claiming a native fallback can repair a poisoned device context.
  native=None
  if local is None:
   native=self.original(x);width=native.shape[-1]//8
   local=native.narrow(-1,self.group.rank_in_group*width,width).contiguous()
  # Every registered/enabled rank participates even after a local layout,
  # pointer, stream, version, workspace or vendor-status rejection.
  assembled=self.group.all_gather(local,dim=-1)
  if self.mode=='qualified':return assembled
  if native is None:native=self.original(x)
  return torch.where((assembled.view(torch.int16)==native.view(torch.int16)).all(),assembled,native)

 def retire(self,*,graphs_destroyed=False):
  """Keep descriptors/workspaces alive for the entire captured-graph lifetime."""
  if not graphs_destroyed:raise RuntimeError('destroy dependent graphs before retiring the router')
  if self.closed:return
  torch.cuda.synchronize(self.original.weight.device)
  for entry in self.entries.values():
   if entry is not None:entry.plan.close()
  self.entries.clear();self.stream_workspaces.clear();self.closed=True
