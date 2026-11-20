"""Optional helper; does not change dense logits or their local maximum reduction."""
import torch,triton
import triton.language as tl
from packed_sampler_kernel import pack_max_id,select_first_rank

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
