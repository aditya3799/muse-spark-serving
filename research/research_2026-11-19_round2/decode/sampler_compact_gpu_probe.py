"""Compact32-bit native-ID mode: virtual-peer correctness and charged local kernels; excludes collective time."""
import argparse,hashlib,json,statistics
from pathlib import Path
import torch,triton
from packed_sampler import PackedDraftCollective
from packed_sampler_kernel import pack_max_id,select_first_rank
here=Path(__file__).resolve().parent

class VirtualPeers:
 def __init__(self,world):self.world_size=world;self.peer_payload=None;self.calls=0
 def all_gather_into_tensor(self,out,local):
  self.calls+=1;out.copy_(self.peer_payload.flatten());out[:local.numel()].copy_(local)

@torch.no_grad()
def main(seed):
 torch.set_num_threads(1);torch.manual_seed(seed)
 records=[];bench={};doc=dict(scope=__doc__,seed=seed,complete=False,correct=True,records=records,bench=bench,
  transport='No NCCL timing. Correctness-only virtual peer copy; timing excludes all transport copies and charges localpack+select vs originalargmax+gather+copy.',
  source_sha256={p:hashlib.sha256((here/p).read_bytes()).hexdigest() for p in ['packed_sampler.py','packed_sampler_kernel.py','sampler_compact_gpu_probe.py']})
 output=here/f'sampler_compact_gpu_{seed}.json'
 def save():output.write_text(json.dumps(doc,indent=2))
 for dtype in (torch.bfloat16,torch.float16,torch.float32):
  bitdtype=torch.int32 if dtype==torch.float32 else torch.int16
  for world,ns in ((2,[1,17]),(4,[17,257]),(8,[7,28,112,224])):
   group=VirtualPeers(world)
   adapter=PackedDraftCollective(tp_size=world,max_tokens=max(ns),dtype=dtype,device=torch.device('cuda',0),compact=True)
   saved=[]
   for n in ns:
    maxima=torch.randn((world,n),device='cuda',dtype=dtype)
    ids=torch.randint(0,2**32,(world,n),device='cuda',dtype=torch.int64)
    payload=torch.empty((world,n),device='cuda',dtype=torch.int64)
    fullout=torch.full((n+16,),-73713,device='cuda',dtype=torch.int64);out=fullout[:n]
    def refresh(pattern):
     maxima.normal_();ids.random_(0,2**32)
     if pattern=='tie':maxima.fill_(2)
     elif pattern=='signedzero':
      maxima.zero_();maxima[1::2].fill_(-0.)
     elif pattern=='posinf':maxima[:,::2]=float('inf')
     elif pattern=='neginf':maxima.fill_(float('-inf'))
     elif pattern=='nan':maxima[1:,::3]=float('nan')
     elif pattern=='bitpatterns':
      bits=torch.randint(-32768,32768,(world,n),device='cuda',dtype=torch.int32)
      if dtype==torch.float32:bits=bits*65537
      maxima.copy_(bits.to(bitdtype).view(dtype))
     ids[:,0]=torch.arange(world,device='cuda')
     if n>1:ids[:,1]=2**32-1-torch.arange(world,device='cuda')
     payload.copy_(((maxima.view(bitdtype).to(torch.int64)&(0xffffffff if dtype==torch.float32 else 0xffff))<<32)|ids)
    refresh('normal');group.peer_payload=payload
    adapter.run(maxima[0],ids[0],out,group);torch.cuda.synchronize()
    with torch.cuda.graph(graph:=torch.cuda.CUDAGraph()):adapter.run(maxima[0],ids[0],out,group)
    checks=[]
    for pattern in ('normal','tie','signedzero','posinf','neginf','nan','bitpatterns'):
     refresh(pattern);graph.replay();torch.cuda.synchronize()
     expected=ids.gather(0,maxima.argmax(0,keepdim=True)).view(-1)
     rawbits=(adapter.gathered[:world*n].view(world,n)>>32).to(bitdtype)
     exact=torch.equal(out,expected) and torch.equal(rawbits,maxima.view(bitdtype)) and bool((fullout[n:]==-73713).all())
     checks.append(dict(input=pattern,exact=exact))
     if not exact:doc['correct']=False;save();raise AssertionError((dtype,world,n,pattern,out,expected))
    records.append(dict(dtype=str(dtype),world=world,N=n,checks=checks,bounded_uint32_ids=True,guardzone_exact=True))
    saved.append((graph,maxima,ids,payload,fullout,n))
    if dtype==torch.bfloat16 and world==8:
     outputs={arm:torch.empty_like(out) for arm in ('baseline','candidate','control')}
     scratch={arm:(torch.empty((1,n),device='cuda',dtype=torch.int64),torch.empty((1,n),device='cuda',dtype=torch.int64)) for arm in ('baseline','control')}
     def original(arm):
      best,selected=scratch[arm]
      torch.argmax(maxima,dim=0,out=best[0]);torch.gather(ids,0,best,out=selected);outputs[arm].copy_(selected.view(-1))
     def local_candidate():
      pack_max_id[(triton.cdiv(n,128),)](maxima[0],ids[0],adapter.packed,n,128,COMPACT=True)
      select_first_rank[(triton.cdiv(n,32),)](payload,outputs['candidate'],n,world,triton.next_power_of_2(world),adapter.value_dtype,32,COMPACT=True,num_warps=4)
     calls=dict(baseline=lambda:original('baseline'),candidate=local_candidate,control=lambda:original('control'))
     timers={};times={a:[] for a in calls}
     for arm,call in calls.items():
      call();torch.cuda.synchronize()
      with torch.cuda.graph(timer:=torch.cuda.CUDAGraph()):
       for _ in range(64):call()
      timers[arm]=timer
     for iteration in range(8):
      refresh('normal' if iteration%2 else 'tie')
      order=list(calls) if iteration%2==0 else list(calls)[::-1]
      for arm in order:
       begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
       begin.record()
       for _ in range(8):timers[arm].replay()
       end.record();end.synchronize();times[arm].append(begin.elapsed_time(end)*1000/512)
      assert all(torch.equal(outputs[a],outputs['baseline']) for a in outputs)
      assert torch.equal((adapter.packed[:n]>>32).to(bitdtype),maxima[0].view(bitdtype))
      assert torch.equal(adapter.packed[:n]&0xffffffff,ids[0])
     median={a:statistics.median(t) for a,t in times.items()}
     bench[str(n)]=dict(times=times,median_us=median,local_candidate_minus_native_us=median['candidate']-median['baseline'])
     print('charged local kernels',n,median,flush=True)
     del timers,timer,calls,scratch,outputs
   # A-B-A graph replay uses the same adapter buffers at changing capturedN.
   for graph,maxima,ids,payload,fullout,n in saved[::-1]+saved:
    maxima.normal_();payload.copy_(((maxima.view(bitdtype).to(torch.int64)&(0xffffffff if dtype==torch.float32 else 0xffff))<<32)|ids)
    graph.replay();torch.cuda.synchronize()
    assert torch.equal(fullout[:n],ids.gather(0,maxima.argmax(0,keepdim=True)).view(-1))
    assert bool((fullout[n:]==-73713).all())
   print('PASS',str(dtype),'TP',world,'N',ns,flush=True)
   del saved,adapter
 doc['complete']=True;save()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=9817351);main(p.parse_args().seed)
