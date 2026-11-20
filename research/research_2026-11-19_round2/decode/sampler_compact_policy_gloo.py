"""Real Gloo compact-range startup agreement; mocked allocation eligibility, no CUDA math claim."""
import gc,json,os,socket,weakref
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1'
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import packed_sampler as runtime
from types import SimpleNamespace as S
here=Path(__file__).resolve().parent

def peer(rank,world,port):
 torch.set_num_threads(1);dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',world_size=world,rank=rank)
 records=[];group=S(world_size=world,cpu_group=dist.group.WORLD)
 original_eligible=runtime._eligible;original_class=runtime.PackedDraftCollective
 runtime._architecture=lambda device:(9,0)
 try:
  for case in ('off','enabled','local_device_ordinal','flag_disagreement','dtype_disagreement','capacity_disagreement','one_unsupported','one_allocation_oom','budget','bounded_native','upper32_edge','one_missing_bound','negative_bound','too_large_bound','noninteger_bound'):
   os.environ['SGLANG_EXPERIMENTAL_DFLASH_PACKED_ARGMAX']='0' if case=='off' or (case=='flag_disagreement' and rank==world-1) else '1'
   runtime._eligible=lambda dtype,device:case!='one_unsupported' or rank!=world-1
   allocated=[]
   class Probe:pass
   def allocate(**kwargs):
    if rank==world-1 and case=='one_allocation_oom':raise torch.OutOfMemoryError('injected recoverable startup allocation failure')
    value=Probe();value.kwargs=kwargs;allocated.append(weakref.ref(value));return value
   runtime.PackedDraftCollective=allocate
   cap=224 if not (case=='capacity_disagreement' and rank==world-1) else 225
   if case=='budget':cap=2**25
   dtype=torch.float16 if case=='dtype_disagreement' and rank==world-1 else torch.bfloat16
   device=f'cuda:{rank}' if case=='local_device_ordinal' else 'cpu'
   bounds=None
   bounded_cases=('bounded_native','upper32_edge','one_missing_bound','negative_bound','too_large_bound','noninteger_bound')
   if case in bounded_cases:bounds=(rank*100,rank*100+17)
   if case=='upper32_edge' and rank==world-1:bounds=(2**32-17,2**32)
   if case=='one_missing_bound' and rank==world-1:bounds=None
   if case=='negative_bound' and rank==world-1:bounds=(-1,17)
   if case=='too_large_bound' and rank==world-1:bounds=(2**32-1,2**32+17)
   if case=='noninteger_bound' and rank==world-1:bounds=(0,1.5)
   result=runtime.maybe_prepare_packed(tp_group=group,max_tokens=cap,dtype=dtype,device=device,native_id_range=bounds)
   expected=case in ('enabled','local_device_ordinal')+bounded_cases;assert (result is not None)==expected,(rank,case)
   outcome=[None]*world;dist.all_gather_object(outcome,result is not None)
   assert all(v==expected for v in outcome)
   modes=[None]*world;dist.all_gather_object(modes,None if result is None else result.kwargs['compact'])
   expected_compact=(case in ('bounded_native','upper32_edge')) if expected else None
   assert all(x==expected_compact for x in modes),(case,modes)
   del result;gc.collect();assert all(ref() is None for ref in allocated)
   if rank==0:records.append(dict(case=case,all_ranks_enabled=outcome,expected=expected,temporary_allocations_released=True,all_ranks_compact=modes))
  if rank==0:(here/f'sampler_compact_policy_tp{world}.json').write_text(json.dumps(dict(scope=__doc__,passed=True,cases=records),indent=2))
 finally:dist.destroy_process_group()

if __name__=='__main__':
 torch.set_num_threads(1)
 for world in (2,4):
  with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
  mp.start_processes(peer,args=(world,port),nprocs=world,join=True,start_method='fork')
  print('PASS startup-policy GlooTP',world,flush=True)
