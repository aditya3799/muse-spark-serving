import ctypes,json,pathlib,statistics,time
import torch
O=pathlib.Path('D:/OpenKernelChallenge/serving/research_2026-11-20/rawscale_gpu')
libs=[ctypes.CDLL(str(O/name)) for name in ('rawscale.dll','specialized.dll')]
funcs=[]
for lib in libs:
 f=lib.pack_raw;f.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p];f.restype=ctypes.c_int;funcs.append(f)
orders=[('baseline','candidate','control'),('control','candidate','baseline'),('candidate','baseline','control'),('control','baseline','candidate'),('baseline','control','candidate'),('candidate','control','baseline')]
report={'scope':'Paired standalone GPU byte-layout conversion. Baseline is newly written generic converter, NOT scored Humming or stock serving. No serving speedup claimed.','cases':[],'complete':False}
for E,N in [(1,4096),(1,8192),(2,4096),(2,8192)]:
 size=E*N*4096*9//16;x=torch.randint(0,256,(size,),device='cuda',dtype=torch.uint8)
 ys={arm:torch.empty_like(x) for arm in orders[0]};samples={arm:[] for arm in ys};wall={arm:[] for arm in ys}
 for rep in range(-2,18):
  if rep%3==0:x.random_(0,256)
  for arm in orders[rep%6]:
   f=funcs[int(arm=='candidate')];a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   torch.cuda.synchronize();start=time.perf_counter_ns();a.record();assert f(x.data_ptr(),ys[arm].data_ptr(),E,N,256,torch.cuda.current_stream().cuda_stream)==0;b.record();b.synchronize()
   duration=(time.perf_counter_ns()-start)/1000
   if rep>=0:samples[arm].append(a.elapsed_time(b)*1000);wall[arm].append(duration)
  assert torch.equal(ys['baseline'],ys['candidate']) and torch.equal(ys['baseline'],ys['control'])
 ratios=[min(a,c)/b for a,b,c in zip(samples['baseline'],samples['candidate'],samples['control'])]
 controls=[a/c for a,c in zip(samples['baseline'],samples['control'])]
 case={'experts':E,'N':N,'bytes':size,'samples_us':samples,'wall_us':wall,'median_conversion_speedup':statistics.median(ratios),'minimum_pair_gain':min(ratios),'control_ratio_range':[min(controls),max(controls)],'exact_each_iteration':True}
 report['cases'].append(case);print({k:v for k,v in case.items() if k not in ('samples_us','wall_us')},flush=True)
 (O/'paired.json').write_text(json.dumps(report,indent=2));del x,ys;torch.cuda.empty_cache()
report['complete']=True;(O/'paired.json').write_text(json.dumps(report,indent=2))
