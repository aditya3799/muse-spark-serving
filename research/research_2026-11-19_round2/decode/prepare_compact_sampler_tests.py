"""Derive separately named compact-mode fixtures; retain measured full-ID files."""
import ast
from pathlib import Path
here=Path(__file__).resolve().parent
s=(here/'sampler_gpu_probe.py').read_text()
def change(a,b,count=None):
 global s
 if count is not None:assert s.count(a)==count,(a,s.count(a))
 else:assert a in s,a
 s=s.replace(a,b)
change('Virtual-peer correctness and charged local kernels; excludes collective time.','Compact32-bit native-ID mode: virtual-peer correctness and charged local kernels; excludes collective time.',1)
change("sampler_gpu_{seed}.json","sampler_compact_gpu_{seed}.json",1)
change("'sampler_gpu_probe.py'","'sampler_compact_gpu_probe.py'",1)
change("device=torch.device('cuda',0))","device=torch.device('cuda',0),compact=True)",1)
change("torch.randint(-(2**62),2**62,(world,n)","torch.randint(0,2**32,(world,n)",1)
change("torch.empty((world,n,2),device='cuda',dtype=torch.int64)","torch.empty((world,n),device='cuda',dtype=torch.int64)",1)
change("ids.random_(-(2**62),2**62)","ids.random_(0,2**32)",1)
change("torch.iinfo(torch.int64).min+torch.arange(world,device='cuda')","torch.arange(world,device='cuda')",1)
change("torch.iinfo(torch.int64).max-torch.arange(world,device='cuda')","2**32-1-torch.arange(world,device='cuda')",1)
change("payload[:,:,0].copy_(maxima.view(bitdtype));payload[:,:,1].copy_(ids)","payload.copy_(((maxima.view(bitdtype).to(torch.int64)&(0xffffffff if dtype==torch.float32 else 0xffff))<<32)|ids)",2)
change("adapter.gathered[:world*2*n].view(world,n,2)[:,:,0].to(bitdtype)","(adapter.gathered[:world*n].view(world,n)>>32).to(bitdtype)",1)
change("full_int64_ids=True","bounded_uint32_ids=True",1)
change("adapter.packed,n,128)","adapter.packed,n,128,COMPACT=True)",1)
change("adapter.value_dtype,32,num_warps=4)","adapter.value_dtype,32,COMPACT=True,num_warps=4)",1)
change("adapter.packed[:2*n].view(n,2)[:,0].to(bitdtype)","(adapter.packed[:n]>>32).to(bitdtype)",1)
change("adapter.packed[:2*n].view(n,2)[:,1]","adapter.packed[:n]&0xffffffff",1)
ast.parse(s);(here/'sampler_compact_gpu_probe.py').write_bytes(s.encode())
s=(here/'sampler_gloo_probe.py').read_text()
change('Real2/4rank Gloo byte/ID parity; does not benchmark NCCL or GPU code.','Real2/4rank Gloo compact32-ID byte parity; no NCCL timing or GPU code.',1)
change("sampler_gloo_tp{world}.json","sampler_compact_gloo_tp{world}.json",1)
change("torch.randint(-(2**62),2**62,(n,)","torch.randint(0,2**32,(n,)",1)
change("torch.iinfo(torch.int64).min+rank","rank",1)
change("torch.iinfo(torch.int64).max-rank","2**32-1-rank",1)
before="""     local=torch.empty((n,2),dtype=torch.int64)
     local[:,0].copy_(values.view(bitdtype));local[:,1].copy_(ids)
     packed=torch.empty(world*n*2,dtype=torch.int64)
     dist.all_gather_into_tensor(packed,local.flatten())
     matrix=packed.view(world,n,2)
     restored=matrix[:,:,0].to(bitdtype).contiguous().view(dtype)"""
after="""     local=((values.view(bitdtype).to(torch.int64)&(0xffffffff if dtype==torch.float32 else 0xffff))<<32)|ids
     packed=torch.empty(world*n,dtype=torch.int64)
     dist.all_gather_into_tensor(packed,local)
     matrix=packed.view(world,n)
     restored=(matrix>>32).to(bitdtype).contiguous().view(dtype)"""
change(before,after,1)
change("matrix[:,:,1]","matrix&0xffffffff",2)
# Parentheses required when the decoded integer tensor is a function argument.
change("torch.equal(matrix&0xffffffff,allids", "torch.equal(matrix&0xffffffff,allids",1)
change("full_int64_ids_exact=True","bounded_uint32_ids_exact=True",1)
ast.parse(s);(here/'sampler_compact_gloo_probe.py').write_bytes(s.encode())
print('prepared compact GPU and real Gloo fixtures')
s=(here/'sampler_policy_gloo.py').read_text()
change('Real Gloo startup agreement; mocked allocation eligibility, no CUDA math claim.','Real Gloo compact-range startup agreement; mocked allocation eligibility, no CUDA math claim.',1)
change("sampler_policy_tp{world}.json","sampler_compact_policy_tp{world}.json",1)
change("'one_allocation_oom','budget'):","'one_allocation_oom','budget','bounded_native','upper32_edge','one_missing_bound','negative_bound','too_large_bound','noninteger_bound'):",1)
change("   result=runtime.maybe_prepare_packed(tp_group=group,max_tokens=cap,dtype=dtype,device=device)",'''   bounds=None
   bounded_cases=('bounded_native','upper32_edge','one_missing_bound','negative_bound','too_large_bound','noninteger_bound')
   if case in bounded_cases:bounds=(rank*100,rank*100+17)
   if case=='upper32_edge' and rank==world-1:bounds=(2**32-17,2**32)
   if case=='one_missing_bound' and rank==world-1:bounds=None
   if case=='negative_bound' and rank==world-1:bounds=(-1,17)
   if case=='too_large_bound' and rank==world-1:bounds=(2**32-1,2**32+17)
   if case=='noninteger_bound' and rank==world-1:bounds=(0,1.5)
   result=runtime.maybe_prepare_packed(tp_group=group,max_tokens=cap,dtype=dtype,device=device,native_id_range=bounds)''',1)
change("expected=case in ('enabled','local_device_ordinal')", "expected=case in ('enabled','local_device_ordinal')+bounded_cases",1)
change("   del result;gc.collect();assert all(ref() is None for ref in allocated)",'''   modes=[None]*world;dist.all_gather_object(modes,None if result is None else result.kwargs['compact'])
   expected_compact=(case in ('bounded_native','upper32_edge')) if expected else None
   assert all(x==expected_compact for x in modes),(case,modes)
   del result;gc.collect();assert all(ref() is None for ref in allocated)''',1)
change("temporary_allocations_released=True)","temporary_allocations_released=True,all_ranks_compact=modes)",1)
ast.parse(s);(here/'sampler_compact_policy_gloo.py').write_bytes(s.encode())
