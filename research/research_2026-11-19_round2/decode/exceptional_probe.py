"""Changed graph inputs, BF16 exceptional values and reserved padding; no timing."""
import hashlib,json
from pathlib import Path
import torch
from prologue_package import Fa3Prologue
from guard_probe import native,store_cache,SimpleNamespace
here=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.manual_seed(7195271);torch.set_num_threads(2)
 phase=torch.arange(8192,device='cuda').float()[:,None]*500000.**(-torch.arange(0,64,2,device='cuda').float()/64)
 rope=torch.cat([phase.cos(),phase.sin()],-1).contiguous()
 norm=SimpleNamespace(_head_dim=64,_num_q_heads=16,_num_kv_heads=2,_eps=1e-5)
 records=[]
 for m in (7,56,57,256):
  for sliding in (False,True):
   x=torch.empty((m,2320),device='cuda',dtype=torch.bfloat16)[:,:2304];x.normal_()
   pos=torch.randint(0,8192,(m,),device='cuda');pos[0]=8191
   p,r=(pos,rope) if sliding else (None,None)
   loc=torch.randperm(3*m,device='cuda')[:m]+1;loc[::3]=0
   initial=[torch.randint(0,256,(3*m+1,128),device='cuda',dtype=torch.uint8) for _ in range(2)]
   actual=[v.clone().view(torch.float8_e4m3fn) for v in initial]
   adapter=Fa3Prologue();reference=lambda x,p,r:native(norm,x,p,r,kv_store=None)
   kw=dict(reference=reference,reference_is_triton=True,compatible_pool=True)
   assert adapter.try_run(x,p,r,*actual,loc,**kw) is not None
   with torch.cuda.graph(graph:=torch.cuda.CUDAGraph()):q8=adapter.try_run(x,p,r,*actual,loc,**kw)
   for mode in ('signed_zero','subnormal','large_finite','all_bf16_words','nan_inf_value_only'):
    if mode=='signed_zero':
     words=(torch.arange(x.numel(),device='cuda')%2*32768).short();x.copy_(words.view(torch.bfloat16).reshape_as(x))
    elif mode=='subnormal':
     words=(torch.arange(x.numel(),device='cuda')%127+1).short();x.copy_(words.view(torch.bfloat16).reshape_as(x))
    elif mode=='large_finite':
     words=torch.where(torch.arange(x.numel(),device='cuda')%2==0,32639,65407).short();x.copy_(words.view(torch.bfloat16).reshape_as(x))
    elif mode=='all_bf16_words':
     words=(torch.arange(x.numel(),device='cuda')%65536).short();x.copy_(words.view(torch.bfloat16).reshape_as(x))
    else:
     x.normal_();words=torch.tensor([0x7f80,0xff80,0x7fc1,0xffc1,0x8000,0],device='cuda',dtype=torch.int32).short()
     x[:,1152:1280].copy_(words.repeat((m*128+5)//6)[:m*128].view(torch.bfloat16).reshape(m,128))
    pos.random_(0,8192);pos[0]=8191
    loc.copy_(torch.randperm(3*m,device='cuda')[:m]+1);loc[::3]=0
    for a,b in zip(actual,initial):a.view(torch.uint8).copy_(b)
    graph.replay();torch.cuda.synchronize()
    q,k=reference(x,p,r);gold=[a.clone() for a in initial]
    store_cache(k.to(torch.float8_e4m3fn).view(torch.uint8),x[:,1152:1280].to(torch.float8_e4m3fn).view(torch.uint8),*gold,loc,row_bytes=128,size_limit=3*m+1)
    checks=dict(q=torch.equal(q8.view(torch.uint8),q.to(torch.float8_e4m3fn).view(torch.uint8)),
     k=torch.equal(actual[0].view(torch.uint8),gold[0]),v=torch.equal(actual[1].view(torch.uint8),gold[1]),
     reserved=all(torch.equal(a[0].view(torch.uint8),b[0]) for a,b in zip(actual,initial)))
    records.append(dict(M=m,sliding=sliding,input=mode,checks=checks))
    print(m,sliding,mode,checks,flush=True)
    if not all(checks.values()):break
 doc=dict(scope=__doc__,records=records,passed=len(records)==40 and all(all(x['checks'].values()) for x in records),
  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),kernel_sha256=hashlib.sha256((here/'prologue_package/kernel.py').read_bytes()).hexdigest())
 (here/'exceptional_probe.json').write_text(json.dumps(doc,indent=2))
 assert doc['passed']
if __name__=='__main__':main()
