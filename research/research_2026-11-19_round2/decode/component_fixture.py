"""Isolated SM86 exact prologue test, charged native casts/store, no attention claim."""
import argparse,hashlib,json,statistics,subprocess,time,sys,importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch,triton
from prologue_package.kernel import conversion_probe
from prologue_package import Fa3Prologue
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('prologue_native_reader',ROOT/'research_2026-11-18_rl_prefill/decode_qkv/bench.py')
reader=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reader;spec.loader.exec_module(reader)
reader.bootstrap(ROOT/'sglang')
native_norm=reader.qk(ROOT/'sglang/python/sglang/kernels/ops/attention/muse_spark_qk_norm_rope_store.py')
_triton_config=native_norm.__globals__['_triton_config']
from sglang.kernels.ops.kvcache.kvcache import store_cache
torch.set_num_threads(2)
@torch.no_grad()
def main(a):
 gen=torch.Generator(device='cuda').manual_seed(a.seed)
 # Extreme casts are checked independently before any fused kernel is trusted.
 edge=torch.arange(65536,device='cuda',dtype=torch.int32).short().view(torch.bfloat16)
 converted=torch.empty(edge.shape,dtype=torch.uint8,device='cuda')
 conversion_probe[(512,)](edge,converted,edge.numel())
 reference=edge.to(torch.float8_e4m3fn).view(torch.uint8)
 cast_exact=torch.equal(converted,reference)
 doc=dict(scope=__doc__,seed=a.seed,device=torch.cuda.get_device_name(),cast_exact=cast_exact,
  bf16_patterns=65536,cast_mismatches=int((reference!=converted).sum()),
  first_mismatched_patterns=torch.where(reference!=converted)[0][:16].tolist(),cases=[],complete=False,serving_qualified=False,
  source_sha256=hashlib.sha256((HERE/'prologue_package/kernel.py').read_bytes()).hexdigest(),
  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),timing='64 original operation calls captured in each timing graph,8 timing graph repeats')
 path=HERE/f'component_{a.seed}.json'
 def save():path.write_text(json.dumps(doc,indent=2))
 save()
 if not cast_exact:print('CAST REJECT');return
 norm=SimpleNamespace(_head_dim=64,_num_q_heads=16,_num_kv_heads=2,_eps=1e-5)
 adapter=Fa3Prologue()
 phase=torch.arange(8192,device='cuda').float()[:,None]*500000.**(-torch.arange(0,64,2,device='cuda').float()/64)
 rope=torch.cat([phase.cos(),phase.sin()],-1).contiguous()
 for m in (8,32,56,57,128,256):
  for sliding in (False,True):
   qkv=torch.randn((m,2304),device='cuda',dtype=torch.bfloat16,generator=gen)
   positions=torch.randint(0,8192,(m,),device='cuda',generator=gen)
   capacity=3*m+1
   loc=(torch.randperm(capacity-1,device='cuda',generator=gen)[:m]+1).long()
   loc[::9]=0
   pools={arm:[torch.zeros((capacity,128),device='cuda',dtype=torch.uint8) for _ in range(2)] for arm in ('native','candidate','control')}
   outputs={};block,warps=_triton_config(m*18)
   candidate_q=torch.empty((m,1024),device='cuda',dtype=torch.uint8)
   def native(arm):
    q,k=native_norm(norm,qkv,positions if sliding else None,rope if sliding else None,kv_store=None)
    q8=q.to(torch.float8_e4m3fn).view(torch.uint8)
    k8=k.to(torch.float8_e4m3fn).view(torch.uint8)
    v8=qkv[:,1152:1280].to(torch.float8_e4m3fn).view(torch.uint8)
    store_cache(k8,v8,*pools[arm],loc,row_bytes=128,size_limit=capacity,reserved_skip_index=0)
    outputs[arm]=q8
   def candidate():
    result=adapter.try_run(qkv,positions if sliding else None,rope if sliding else None,
      pools['candidate'][0].view(torch.float8_e4m3fn),pools['candidate'][1].view(torch.float8_e4m3fn),loc,
      reference=lambda x,p,r:native_norm(norm,x,p,r,kv_store=None),reference_is_triton=True,compatible_pool=True)
    assert result is not None,adapter.last_reason
    outputs['candidate']=result.view(torch.uint8)
   calls={'native':lambda:native('native'),'control':lambda:native('control'),'candidate':candidate}
   graphs={}
   for arm,call in calls.items():
    for _ in range(2):call()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph:=torch.cuda.CUDAGraph()):call()
    graphs[arm]=graph
   timers={}
   for arm,graph in graphs.items():
    original_output=outputs[arm]
    with torch.cuda.graph(timer:=torch.cuda.CUDAGraph()):
     for _ in range(64):calls[arm]()
    timers[arm]=timer
    outputs[arm]=original_output
   rec=dict(M=m,sliding=sliding,block_m=block,checks=[],times={arm:[] for arm in calls});doc['cases'].append(rec)
   for iteration in range(8):
    if iteration%3==0:qkv.copy_(torch.randn(qkv.shape,device='cuda',dtype=qkv.dtype,generator=gen))
    elif iteration%3==1:qkv.copy_(torch.randint(-8,9,qkv.shape,device='cuda',generator=gen).bfloat16())
    else:qkv.mul_(-16)
    # Refresh actual positions and sparse physical/SWA destinations in graph-owned storage.
    positions.copy_(torch.randint(0,8192,(m,),device='cuda',generator=gen))
    loc.copy_((torch.randperm(capacity-1,device='cuda',generator=gen)[:m]+1).long());loc[::9]=0
    for arm in graphs:graphs[arm].replay()
    torch.cuda.synchronize()
    parity={arm:dict(q=torch.equal(outputs[arm],outputs['native']),
      k=torch.equal(pools[arm][0],pools['native'][0]),v=torch.equal(pools[arm][1],pools['native'][1]),
      slot0=all(bool((x[0]==0).all()) for x in pools[arm])) for arm in ('candidate','control')}
    rec['checks'].append(parity)
    if not all(all(v.values()) for v in parity.values()):doc['rejected']=dict(M=m,sliding=sliding,iteration=iteration);save();print('REJECT',doc['rejected'],parity,flush=True);return
    order=['native','candidate','control'] if iteration%2==0 else ['control','candidate','native']
    for arm in order:
     begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
     begin.record()
     for _ in range(8):timers[arm].replay()
     end.record();end.synchronize();rec['times'][arm].append(begin.elapsed_time(end)*1000/512)
   rec['median_speedup']=statistics.median(rec['times']['native'])/statistics.median(rec['times']['candidate'])
   rec['median_control_ratio']=statistics.median(rec['times']['native'])/statistics.median(rec['times']['control'])
   doc['qualification']=adapter.receipts;save();print(m,sliding,'PASS',round(rec['median_speedup'],4),round(rec['median_control_ratio'],4),flush=True)
 doc['complete']=True;doc['qualification']=adapter.receipts;doc['peak_allocated_bytes']=torch.cuda.max_memory_allocated();save()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=7193851);main(p.parse_args())
