"""Correctness-only reduction edge cases, especially subnormal comparisons."""
import torch
from packed_sampler import PackedDraftCollective

class PeerData:
 world_size=8
 def all_gather_into_tensor(self,out,local):out.copy_(self.payload.flatten());out[:local.numel()].copy_(local)

@torch.no_grad()
def run():
 records=[];world=8;n=17
 for dtype in (torch.bfloat16,torch.float16,torch.float32):
  integer=torch.int32 if dtype==torch.float32 else torch.int16
  sign=0x80000000 if dtype==torch.float32 else 0x8000
  nan=0x7fa00001 if dtype==torch.float32 else (0x7f81 if dtype==torch.bfloat16 else 0x7c01)
  base=torch.arange(1,world+1,device='cuda',dtype=torch.int64)[:,None].expand(world,n).contiguous()
  for compact in (False,True):
   maxima=torch.empty((world,n),device='cuda',dtype=dtype)
   ids=torch.arange(world*n,device='cuda',dtype=torch.int64).reshape(world,n)
   if compact:ids[:,0]=2**32-1-torch.arange(world,device='cuda')
   else:
    ids[:,0]=torch.iinfo(torch.int64).max-torch.arange(world,device='cuda')
    ids[:,1]=torch.iinfo(torch.int64).min+torch.arange(world,device='cuda')
   group=PeerData();group.payload=torch.empty((world,n) if compact else (world,n,2),device='cuda',dtype=torch.int64)
   helper=PackedDraftCollective(tp_size=world,max_tokens=n,dtype=dtype,device=maxima.device,compact=compact)
   out=torch.empty(n,device='cuda',dtype=torch.int64)
   for pattern,bits in [('positive_subnormal',base),('negative_subnormal',base|sign),('signed_zero',(base%2)*sign),('all_nan_payloads',base|nan),('negative_nan_payloads',base|nan|sign)]:
    maxima.copy_(bits.to(integer).view(dtype))
    if compact:group.payload.copy_(((maxima.view(integer).to(torch.int64)&(0xffffffff if dtype==torch.float32 else 0xffff))<<32)|ids)
    else:group.payload[:,:,0].copy_(maxima.view(integer));group.payload[:,:,1].copy_(ids)
    helper.run(maxima[0],ids[0],out,group);torch.cuda.synchronize()
    expected=ids.gather(0,maxima.argmax(0,keepdim=True)).view(-1)
    restored=(helper.gathered.view(world,n)>>32).to(integer) if compact else helper.gathered.view(world,n,2)[:,:,0].to(integer)
    exact=torch.equal(out,expected) and torch.equal(restored,maxima.view(integer))
    records.append(dict(dtype=str(dtype),compact=compact,pattern=pattern,exact=exact))
    assert exact,(dtype,compact,pattern,out,expected)
 return records
