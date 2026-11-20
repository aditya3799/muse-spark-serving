"""Guarded-prologue rejection, graph-fallback and storage-contract checks."""
import hashlib,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from prologue_package import Fa3Prologue
here=Path(__file__).resolve().parent;root=here.parents[1]
spec=importlib.util.spec_from_file_location('guard_native_reader',root/'research_2026-11-18_rl_prefill/decode_qkv/bench.py')
reader=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reader;spec.loader.exec_module(reader)
reader.bootstrap(root/'sglang')
native=reader.qk(root/'sglang/python/sglang/kernels/ops/attention/muse_spark_qk_norm_rope_store.py')
from sglang.kernels.ops.kvcache.kvcache import store_cache

@torch.no_grad()
def main():
 torch.manual_seed(6519371);m=56;capacity=173
 qkv=torch.randn((m,2304),device='cuda',dtype=torch.bfloat16)
 pos=torch.randint(0,4096,(m,),device='cuda')
 phase=torch.arange(4096,device='cuda').float()[:,None]*500000.**(-torch.arange(0,64,2,device='cuda').float()/64)
 rope=torch.cat([phase.cos(),phase.sin()],-1).contiguous()
 loc=torch.randperm(capacity-1,device='cuda')[:m].long()+1;loc[::7]=0
 pools=[torch.randint(256,(capacity,128),device='cuda',dtype=torch.uint8).view(torch.float8_e4m3fn) for _ in range(2)]
 pristine=[x.clone() for x in pools]
 norm=SimpleNamespace(_head_dim=64,_num_q_heads=16,_num_kv_heads=2,_eps=1e-5)
 def reference(x,p,r):return native(norm,x,p,r,kv_store=None)
 args=[qkv,pos,rope,*pools,loc]
 kw=dict(reference=reference,reference_is_triton=True,compatible_pool=True)
 records=[];adapter=Fa3Prologue()
 tests=[('scale',{},dict(k_scale=2.)),('explicit_unit_scale',{},dict(k_scale=1.)),
        ('wrong_backend',{},dict(reference_is_triton=False)),('pool_owner',{},dict(compatible_pool=False)),
        ('asymmetric_geometry',{},dict(num_kv_heads=1)),('value_width',{4:pools[1][:,:64]},{}),
        ('input_dtype',{0:qkv.float()},{}),('input_feature_stride',{0:torch.empty((m,4608),device='cuda',dtype=torch.bfloat16)[:,::2]},{}),
        ('cache_stride',{3:torch.empty((capacity,256),device='cuda',dtype=torch.uint8)[:,::2].view(torch.float8_e4m3fn)},{}),
        ('location_stride',{5:torch.empty(m*2,device='cuda',dtype=torch.int64)[::2]},{}),
        ('position_stride',{1:torch.empty(m*2,device='cuda',dtype=torch.int64)[::2]},{}),
        ('rope_dtype',{2:rope.bfloat16()},{}),('epsilon',{},dict(eps=float('nan')))]
 for name,changes,overrides in tests:
  current=list(args)
  for i,value in changes.items():current[i]=value
  out=adapter.try_run(*current,**{**kw,**overrides})
  untouched=all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(pools,pristine))
  assert out is None and untouched,(name,adapter.last_reason)
  records.append(dict(case=name,rejected=True,reason=adapter.last_reason,full_pools_unchanged=True))
 # A true noncontiguous row-major input retains the original inner-feature order.
 strided=torch.empty((m,2320),device='cuda',dtype=torch.bfloat16)[:,:2304];strided.copy_(qkv)
 candidate=adapter.try_run(strided,pos,rope,*pools,loc,**kw)
 assert candidate is not None
 rq,rk=reference(strided,pos,rope)
 gold=[x.clone() for x in pristine]
 store_cache(rk.to(torch.float8_e4m3fn).view(torch.uint8),strided[:,1152:1280].to(torch.float8_e4m3fn).view(torch.uint8),
             gold[0].view(torch.uint8),gold[1].view(torch.uint8),loc,row_bytes=128,size_limit=capacity)
 assert torch.equal(candidate.view(torch.uint8),rq.to(torch.float8_e4m3fn).view(torch.uint8))
 assert all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(pools,gold))
 records.append(dict(case='row_stride_supported',exact=True,full_pools_exact=True,slot0_preserved=True))
 # Cold graph capture must execute the original complete path, not qualify/write scratch.
 cold=Fa3Prologue();capture_outputs=[]
 def fallback():
  out=cold.try_run(*args,**kw)
  assert out is None
  q,k=reference(qkv,pos,rope)
  store_cache(k.to(torch.float8_e4m3fn).view(torch.uint8),qkv[:,1152:1280].to(torch.float8_e4m3fn).view(torch.uint8),
              pools[0].view(torch.uint8),pools[1].view(torch.uint8),loc,row_bytes=128,size_limit=capacity)
  return q.to(torch.float8_e4m3fn)
 # Warm exact native contiguous-stride path while keeping the adapter cold.
 warmq,warmk=reference(qkv,pos,rope)
 store_cache(warmk.to(torch.float8_e4m3fn).view(torch.uint8),qkv[:,1152:1280].to(torch.float8_e4m3fn).view(torch.uint8),
             pools[0].view(torch.uint8),pools[1].view(torch.uint8),loc,row_bytes=128,size_limit=capacity)
 warmq.to(torch.float8_e4m3fn);torch.cuda.synchronize()
 before_capture=[x.clone() for x in pools]
 with torch.cuda.graph(graph:=torch.cuda.CUDAGraph()):captured=fallback()
 qkv.normal_();pos.random_(0,4096);graph.replay();torch.cuda.synchronize()
 expected,expected_k=reference(qkv,pos,rope)
 assert torch.equal(captured.view(torch.uint8),expected.to(torch.float8_e4m3fn).view(torch.uint8))
 store_cache(expected_k.to(torch.float8_e4m3fn).view(torch.uint8),qkv[:,1152:1280].to(torch.float8_e4m3fn).view(torch.uint8),
             before_capture[0].view(torch.uint8),before_capture[1].view(torch.uint8),loc,row_bytes=128,size_limit=capacity)
 assert all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(pools,before_capture))
 records.append(dict(case='cold_capture_fallback',exact=True,full_pools_exact=True,reason=cold.last_reason,qualification_count=len(cold.receipts)))
 # Deliberate failed proof may modify only scratch, never the caller's backing pools.
 failed=Fa3Prologue();before=[x.clone() for x in pools]
 def bad_reference(x,p,r):
  q,k=reference(x,p,r);return q+1,k
 out=failed.try_run(*args,**{**kw,'reference':bad_reference})
 assert out is None and all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(pools,before))
 records.append(dict(case='failed_shadow_proof',reason=failed.last_reason,full_pools_unchanged=True))
 capped=Fa3Prologue(calibration_budget_bytes=1)
 assert capped.try_run(*args,**kw) is None
 records.append(dict(case='scratch_budget',reason=capped.last_reason))
 doc=dict(scope=__doc__,device=torch.cuda.get_device_name(),passed=True,records=records,
          qualification=adapter.receipts,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (here/'guard_probe.json').write_text(json.dumps(doc,indent=2));print('PASS',len(records),'guard/lifetime controls',flush=True)
if __name__=='__main__':main()
