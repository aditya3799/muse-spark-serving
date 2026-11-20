"""CABI exact coefficient gate and local isolated dequant timing, no serving claim."""
import ctypes as ct
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
here=Path(__file__).resolve().parent
lib=ct.CDLL(str(here/'libregister_dequant_sm86.so'))
lib.dequant_run.argtypes=(ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_void_p)
lib.dequant_run.restype=ct.c_int
fp4=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],device='cuda')
fp8=torch.arange(256,device='cuda').byte().view(torch.float8_e4m3fn).float()
table=(fp8[:,None]*fp4).bfloat16()
report=dict(complete=False,device=torch.cuda.get_device_name(),cases=[],
 source_sha256=hashlib.sha256((here/'register_dequant_probe.cu').read_bytes()).hexdigest(),
 scope='Actual CUDA BF16 coefficient-only oracle; no WGMMA or serving throughput measured')

def call(words,sf,out,variant):
 rc=lib.dequant_run(words.data_ptr(),sf.data_ptr(),table.data_ptr(),out.data_ptr(),len(words),variant,torch.cuda.current_stream().cuda_stream)
 if rc:raise RuntimeError(('cuda',rc))
# Exhaustive all16codes*all256scale bytes, including NaNs, signed0, subnormals.
words=[];scales=[]
for sf in range(256):
 for q in range(16):
  word=0
  for r in range(4):
   for h in range(2):
    for bit,offset in enumerate(([6+3*r,7+3*r,8+3*r,15+3*r] if r<3 else [4,0,1,3])):
     word|=((q>>bit)&1)<<((offset+16*h)%32)
  words.append([word]*4);scales.append(sf|(sf<<8))
words=torch.from_numpy(np.array(words,np.uint32).view(np.int32)).cuda()
scales=torch.from_numpy(np.array(scales,np.uint16).view(np.int16)).cuda()
parent=torch.empty((len(words),16),device='cuda',dtype=torch.int32)
candidate=torch.empty_like(parent)
call(words,scales,parent,0)
for variant in (1,2):
 call(words,scales,candidate,variant);torch.cuda.synchronize()
 bad=int((candidate!=parent).sum());report['cases'].append(dict(kind='all_codes_scales',variant=variant,output_bf16_values=candidate.numel()*2,unequal_words=bad))
 if bad:
  badix=(candidate!=parent).nonzero()[:20]
  report['examples']=[dict(record=int(i),word=int(j),actual=int(candidate[i,j]),expected=int(parent[i,j])) for i,j in badix]
  (here/'register_dequant_gpu.json').write_text(json.dumps(report,indent=2));raise AssertionError((variant,bad))
del words,scales,parent,candidate
for seed in (930269,930281):
 torch.manual_seed(seed);count=1<<18
 words=torch.randint(0,2**32,(count,4),device='cuda',dtype=torch.int64).to(torch.int32)
 scales=torch.randint(0,65536,(count,),device='cuda',dtype=torch.int32).to(torch.int16)
 parent=torch.empty((count,16),device='cuda',dtype=torch.int32);candidate=torch.empty_like(parent)
 call(words,scales,parent,0)
 for variant in (1,2):
  call(words,scales,candidate,variant);torch.cuda.synchronize()
  assert torch.equal(candidate,parent),(seed,variant)
  samples={key:[] for key in ('baseline','candidate','control')}
  for i in range(8):
   order=[('baseline',0),('candidate',variant),('control',0)]
   if i%2:order.reverse()
   for key,which in order:
    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    begin.record();call(words,scales,candidate,which);end.record();end.synchronize()
    samples[key].append(begin.elapsed_time(end)*1000)
  row=dict(kind='random_mixed_signs_and_exceptionals',seed=seed,variant=variant,
           records=count,output_bf16_values=parent.numel()*2,exact=True,samples_us=samples)
  report['cases'].append(row)
 del words,scales,parent,candidate
report.update(complete=True,historical_peak_allocated=torch.cuda.max_memory_allocated())
(here/'register_dequant_gpu.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
