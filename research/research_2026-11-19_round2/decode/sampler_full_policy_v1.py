"""Optional helper; does not change dense logits or their local maximum reduction."""
import logging,os
import torch,triton
import torch.distributed as dist
import triton.language as tl
from packed_sampler_kernel import pack_max_id,select_first_rank
logger=logging.getLogger(__name__)

def _eligible(dtype,device):
 return torch.device(device).type=='cuda' and dtype in (torch.float16,torch.bfloat16,torch.float32)

def _architecture(device):
 return torch.cuda.get_device_capability(device) if torch.device(device).type=='cuda' else None

def maybe_prepare_packed(*,tp_group,max_tokens,dtype,device):
 """Agree once outside capture. No per-rank runtime fallback may skip a collective."""
 if tp_group is None or int(tp_group.world_size)<=1:return None
 # Real GroupCoordinator asserts this communicator exists at construction.
 cpu_group=tp_group.cpu_group
 if cpu_group is None:raise ValueError('multi-rank startup agreement requires CPU group')
 enabled=os.environ.get('SGLANG_EXPERIMENTAL_DFLASH_PACKED_ARGMAX','0')=='1'
 estimate=16*int(max_tokens)*(int(tp_group.world_size)+1)
 descriptor=dict(enabled=enabled,max_tokens=int(max_tokens),dtype=str(dtype),world=int(tp_group.world_size),
  supported=_eligible(dtype,device) and int(max_tokens)>0 and estimate<=64*1024**2,arch=_architecture(device))
 peers=[None]*int(tp_group.world_size);dist.all_gather_object(peers,descriptor,group=cpu_group)
 if not enabled or not all(x==descriptor for x in peers) or not descriptor['supported']:
  logger.info('DFLASH_PACKED_ARGMAX native fallback: disabled or startup descriptors disagree/unsupported')
  return None
 candidate=None;error=None
 try:candidate=PackedDraftCollective(tp_size=tp_group.world_size,max_tokens=max_tokens,dtype=dtype,device=device)
 except torch.OutOfMemoryError as exc:error=type(exc).__name__+': '+str(exc)
 statuses=[None]*int(tp_group.world_size)
 dist.all_gather_object(statuses,dict(ready=candidate is not None,error=error),group=cpu_group)
 if not all(x['ready'] for x in statuses):
  candidate=None
  logger.info('DFLASH_PACKED_ARGMAX native fallback: startup allocation rejected %s',statuses)
  return None
 logger.info('DFLASH_PACKED_ARGMAX enabled tp=%d max_tokens=%d scratch_bytes=%d dtype=%s',tp_group.world_size,max_tokens,estimate,dtype)
 return candidate

class PackedDraftCollective:
 def __init__(self,*,tp_size,max_tokens,dtype,device):
  if dtype not in (torch.float16,torch.bfloat16,torch.float32):raise ValueError('unsupported maximum dtype')
  if tp_size<=1 or max_tokens<=0:raise ValueError('requires multi-rank positive capacity')
  self.tp_size=int(tp_size);self.max_tokens=int(max_tokens);self.dtype=dtype;self.device=torch.device(device)
  self.packed=torch.empty((2*max_tokens,),dtype=torch.int64,device=device)
  self.device=self.packed.device
  self.gathered=torch.empty((tp_size*2*max_tokens,),dtype=torch.int64,device=device)
  self.value_dtype={torch.float16:tl.float16,torch.bfloat16:tl.bfloat16,torch.float32:tl.float32}[dtype]
 def run(self,local_max,local_ids,out,tp_group):
  n=local_max.numel()
  if not (0<n<=self.max_tokens and local_max.dtype==self.dtype and local_ids.dtype==out.dtype==torch.int64):raise ValueError('shape/dtype mismatch')
  if any(x.device!=self.device or not x.is_contiguous() or x.ndim!=1 or x.numel()!=n for x in (local_max,local_ids,out)):raise ValueError('device/layout mismatch')
  if tp_group.world_size!=self.tp_size:raise ValueError('TP group changed')
  packed=self.packed[:2*n];gathered=self.gathered[:self.tp_size*2*n]
  pack_max_id[(triton.cdiv(n,128),)](local_max,local_ids,packed,n,128)
  tp_group.all_gather_into_tensor(gathered,packed)
  select_first_rank[(triton.cdiv(n,32),)](gathered,out,n,self.tp_size,triton.next_power_of_2(self.tp_size),self.value_dtype,32,num_warps=4)
