"""Real2/4rank Gloo compact32-ID byte parity; no NCCL timing or GPU code."""
import datetime,json,os,socket,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1'
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
here=Path(__file__).resolve().parent

def peer(rank,world,port):
 torch.set_num_threads(1)
 dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=world)
 records=[];g=torch.Generator().manual_seed(187391+rank)
 try:
  for dtype in (torch.bfloat16,torch.float16,torch.float32):
   bitdtype=torch.int32 if dtype==torch.float32 else torch.int16
   for n in (1,7,28,56,112,224,257):
    for pattern in ('normal','tie','signedzero','posinf','neginf','firstnan','multinan','bitpatterns'):
     values=torch.randn(n,generator=g).to(dtype)
     if pattern=='tie':values.fill_(3)
     elif pattern=='signedzero':values.fill_(-0. if rank%2 else 0.)
     elif pattern=='posinf':values[::2]=float('inf')
     elif pattern=='neginf':values.fill_(float('-inf'))
     elif pattern=='firstnan':
      if rank==world-1:values[::3]=float('nan')
     elif pattern=='multinan':
      if rank in (0,world-1):values[::3]=float('nan')
     elif pattern=='bitpatterns':
      raw=torch.randint(-32768,32768,(n,),dtype=torch.int32,generator=g)
      if dtype==torch.float32:raw=raw*65537
      values=raw.to(bitdtype).view(dtype)
     ids=torch.randint(0,2**32,(n,),dtype=torch.int64,generator=g)
     ids[0]=rank
     if n>1:ids[1]=2**32-1-rank
     maxima=torch.empty(world*n,dtype=dtype);allids=torch.empty(world*n,dtype=torch.int64)
     dist.all_gather_into_tensor(maxima,values);dist.all_gather_into_tensor(allids,ids)
     best=torch.argmax(maxima.view(world,n),dim=0,keepdim=True)
     expected=torch.gather(allids.view(world,n),0,best).view(-1)
     local=((values.view(bitdtype).to(torch.int64)&(0xffffffff if dtype==torch.float32 else 0xffff))<<32)|ids
     packed=torch.empty(world*n,dtype=torch.int64)
     dist.all_gather_into_tensor(packed,local)
     matrix=packed.view(world,n)
     restored=(matrix>>32).to(bitdtype).contiguous().view(dtype)
     assert torch.equal(restored.view(bitdtype),maxima.view(world,n).view(bitdtype))
     assert torch.equal(matrix&0xffffffff,allids.view(world,n))
     winners=torch.argmax(restored,dim=0,keepdim=True)
     actual=torch.gather(matrix&0xffffffff,0,winners).view(-1)
     assert torch.equal(actual,expected),(rank,world,dtype,n,pattern)
     if rank==0:records.append(dict(dtype=str(dtype),N=n,pattern=pattern,maxbits_exact=True,bounded_uint32_ids_exact=True,tokens_exact=True))
  if rank==0:(here/f'sampler_compact_gloo_tp{world}.json').write_text(json.dumps(dict(scope=__doc__,passed=True,world_size=world,cases=records),indent=2))
 finally:dist.destroy_process_group()

def main():
 torch.set_num_threads(1)
 for world in (2,4):
  with socket.socket() as listener:listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
  start=time.monotonic();mp.start_processes(peer,args=(world,port),nprocs=world,join=True,start_method='fork')
  print('PASS GlooTP',world,'seconds',time.monotonic()-start,flush=True)
if __name__=='__main__':main()
