"""Final router lifecycle checks and exploratory local graph timings."""
import json,statistics
import torch
from validate_router import ROOT,SOURCE,router_class,stock,mm,relative
cls=router_class(SOURCE.read_text(encoding='utf-8'))
torch.manual_seed(31)
r=cls(8192,256,torch.float32).cuda()
with torch.no_grad(): r.weight.normal_()
x=torch.randn(33,8192,device='cuda',dtype=torch.bfloat16)
r.forward_split2(x); initial=r._split2_weight
assert r.forward_split2(x).shape==(33,256) and r._split2_weight is initial
with torch.no_grad(): r.weight.add_(0.125)
out=r.forward_split2(x)
assert r._split2_weight is not initial and relative(out,stock(r,x))<1e-4
second=r._split2_weight
with torch.no_grad(): r.weight.copy_(torch.randn_like(r.weight))
assert relative(r.forward_split2(x),stock(r,x))<1e-4 and r._split2_weight is not second
assert set(r.state_dict())=={'weight','e_score_correction_bias'}
# Replacing the Parameter also rebuilds the pack.
with torch.no_grad(): r.weight=torch.nn.Parameter(torch.randn_like(r.weight),requires_grad=False)
assert relative(r.forward_split2(x),stock(r,x))<1e-4
assert r._split2_weight.numel()*r._split2_weight.element_size()==2*r.weight.numel()*2

# Prepack both versions outside capture. Local timings are not Hopper evidence.
w=r.weight.detach(); hi=w.to(torch.bfloat16); lo=(w-hi.float()).to(torch.bfloat16)
hi=hi.t().contiguous(); lo=lo.t().contiguous(); r._build_split2()
flush=torch.empty(32<<20,device='cuda',dtype=torch.uint8)
records=[]
for m in [16,32,128,256,1024]:
 x=torch.randn(m,8192,device='cuda',dtype=torch.bfloat16)
 def baseline():
  acc=mm(x,hi,out_dtype=torch.float32); acc+=mm(x,lo,out_dtype=torch.float32); return acc
 def candidate(): return r.forward_split2(x)
 graphs=[]; outs=[]
 for fn in [baseline,candidate]:
  stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3): fn()
  torch.cuda.current_stream().wait_stream(stream)
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph): out=fn()
  graphs.append(graph); outs.append(out)
 torch.cuda.synchronize(); assert relative(outs[1],outs[0])<1e-4
 times=[[],[]]
 for rep in range(26):
  for arm in ([0,1] if rep%2==0 else [1,0]):
   flush.zero_()
   start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   start.record(); graphs[arm].replay(); end.record(); end.synchronize()
   if rep>=6: times[arm].append(start.elapsed_time(end)*1000)
 a,b=map(statistics.median,times)
 records.append({'m':m,'stock_us':a,'candidate_us':b,'stock_over_candidate':a/b})
result={'device':torch.cuda.get_device_name(),'lifecycle_checks':['cache reuse','in-place add invalidation','copy_ invalidation','parameter replacement','state_dict unchanged','persistent packed bytes match old hi+lo'],'timing_method':'local RTX3050; paired AB/BA CUDA graphs; 32 MiB flush before each replay; 20 samples/arm; cuBLAS adapter for torch2.6; exploratory, not target-H200 or full-serving evidence','timings':records}
(ROOT/'validation/router_final.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
