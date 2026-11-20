"""Padded-native versus real-row prologue bytes, SM86 only; no attention/timing claim."""
import hashlib,json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
DECODE=HERE.parent/'decode'
sys.path.insert(0,str(DECODE))
import torch
from component_fixture import native_norm,_triton_config,store_cache,SimpleNamespace
from prologue_package import Fa3Prologue

@torch.no_grad()
def main():
 torch.set_num_threads(2)
 started=time.perf_counter();gen=torch.Generator(device='cuda').manual_seed(2611191229)
 adapter=Fa3Prologue();codec=adapter.prepare_codec(torch.device('cuda',0));assert codec
 norm=SimpleNamespace(_head_dim=64,_num_q_heads=16,_num_kv_heads=2,_eps=1e-5)
 phase=torch.arange(32768,device='cuda').float()[:,None]*500000.**(-torch.arange(0,64,2,device='cuda').float()/64)
 rope=torch.cat([phase.cos(),phase.sin()],-1).contiguous();del phase
 records=[]
 for n,padded in [(56,64),(454,512),(753,1024),(3012,4096),(3640,4096),(16383,16384)]:
  for sliding in (False,True):
   stride=2320 if sliding else 2304
   x=torch.empty((padded,stride),device='cuda',dtype=torch.bfloat16)[:,:2304]
   x.normal_(generator=gen)
   pos=torch.randint(0,32768,(padded,),device='cuda',generator=gen)
   cap=3*n+17
   initial=[torch.randint(0,256,(cap,128),device='cuda',dtype=torch.uint8,generator=gen) for _ in range(2)]
   actual=[v.clone().view(torch.float8_e4m3fn) for v in initial]
   loc=torch.randperm(cap-1,device='cuda',generator=gen)[:n]+1;loc[::11]=0
   def candidate():return adapter._launch(x[:n],pos[:n] if sliding else None,rope if sliding else None,*actual,loc,1e-5,codec)
   # Explicit fixture bypass of production per-shape16MiB calibration policy.
   # Actual encoder is qualified; no production shape publication is performed.
   candidate();torch.cuda.synchronize()
   with torch.cuda.graph(graph:=torch.cuda.CUDAGraph()):out=candidate()
   for mutation in ('random','integer','padded_poison'):
    if mutation=='integer':x.random_(-8,9,generator=gen)
    else:x.normal_(generator=gen);x[:,::7].mul_(8)
    if mutation=='padded_poison':x[n:].fill_(float('nan'))
    pos.random_(0,32768,generator=gen);pos[0]=32767
    loc.copy_(torch.randperm(cap-1,device='cuda',generator=gen)[:n]+1);loc[::11]=0
    for a,b in zip(actual,initial):a.view(torch.uint8).copy_(b)
    graph.replay()
    q,k=native_norm(norm,x,pos if sliding else None,rope if sliding else None,kv_store=None)
    gold=[v.clone() for v in initial]
    store_cache(k[:n].to(torch.float8_e4m3fn).view(torch.uint8),x[:n,1152:1280].to(torch.float8_e4m3fn).view(torch.uint8),
       *gold,loc,row_bytes=128,size_limit=cap,reserved_skip_index=0)
    torch.cuda.synchronize()
    checks=dict(q=torch.equal(out.view(torch.uint8),q[:n].to(torch.float8_e4m3fn).view(torch.uint8)),
      k=torch.equal(actual[0].view(torch.uint8),gold[0]),v=torch.equal(actual[1].view(torch.uint8),gold[1]),
      reserved=all(torch.equal(a[0].view(torch.uint8),b[0]) for a,b in zip(actual,initial)))
    records.append(dict(real=n,padded=padded,rope=sliding,stride=stride,mutation=mutation,
      native_block=_triton_config(padded*18)[0],candidate_block=adapter._block(n),checks=checks))
    assert all(checks.values()),records[-1]
    del q,k,gold
   del graph,out,x,pos,initial,actual,loc
   print(n,padded,sliding,'PASS',flush=True)
 # Concrete graph identity control: replacing a tensor object is NOT updating
 # the storage captured earlier. Keep both destinations valid; never illegal IDs.
 n=56;cap=3*n+1
 x=torch.randn((n,2304),device='cuda',dtype=torch.bfloat16,generator=gen)
 old=torch.arange(1,n+1,device='cuda');fresh=torch.arange(n+1,2*n+1,device='cuda')
 pools=[torch.zeros((cap,128),device='cuda',dtype=torch.float8_e4m3fn) for _ in range(2)]
 adapter._launch(x,None,None,*pools,old,1e-5,codec);torch.cuda.synchronize()
 with torch.cuda.graph(graph:=torch.cuda.CUDAGraph()):held=adapter._launch(x,None,None,*pools,old,1e-5,codec)
 for p in pools:p.zero_()
 current_locations=fresh
 graph.replay();torch.cuda.synchronize()
 capture_kept_old_pointer=bool((pools[0][1:n+1].view(torch.uint8)!=0).any()) and not bool((pools[0][n+1:2*n+1].view(torch.uint8)!=0).any())
 assert capture_kept_old_pointer
 for p in pools:p.zero_()
 current=adapter._launch(x,None,None,*pools,current_locations,1e-5,codec);torch.cuda.synchronize()
 eager_used_fresh_pointer=not bool((pools[0][1:n+1].view(torch.uint8)!=0).any()) and bool((pools[0][n+1:2*n+1].view(torch.uint8)!=0).any())
 assert eager_used_fresh_pointer
 doc=dict(scope=__doc__,passed=True,device=torch.cuda.get_device_name(),seed=2611191229,cases=records,
   production_shape_gate_bypassed_only_in_fixture=True,calibration_budget_unchanged=adapter.calibration_budget_bytes,
   metadata_identity=dict(capture_kept_old_pointer=capture_kept_old_pointer,eager_used_fresh_pointer=eager_used_fresh_pointer),
   peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),elapsed_s=time.perf_counter()-started,
   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),kernel_sha256=hashlib.sha256((DECODE/'prologue_package/kernel.py').read_bytes()).hexdigest(),
   limitations=['No native FP8 attention on SM86','No actual SGLang BCG execution','Large prefill production shape qualification remains disabled under current16MiB policy','No speed or serving claim'])
 (HERE/'prefill_prologue_bytes.json').write_text(json.dumps(doc,indent=2))
 print(json.dumps({k:v for k,v in doc.items() if k!='cases'},indent=2))
if __name__=='__main__':main()

