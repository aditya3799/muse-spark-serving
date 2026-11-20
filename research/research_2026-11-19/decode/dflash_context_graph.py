"""Same-shape DFlash context graph, including dynamic metadata staging.

Only orchestration changes. Exact original native projection, norm, rotary and
prefix-valid write functions are used in every arm. No extra speculative writes.
"""
import argparse
import fcntl
import hashlib
import inspect
import json
import statistics
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import sgl_kernel

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'research_2026-11-18_rl_dflash'))
sys.path.insert(0,str(ROOT/'research_2026-11-18_rl'))
import joint_bench as old
from run_pair import telemetry

def run(a,directory):
 rng=torch.Generator(device='cuda').manual_seed(a.seed)
 def rand(shape,scale=1):return torch.empty(shape,device='cuda',dtype=torch.bfloat16).normal_(std=scale,generator=rng)
 fc=rand((8192,40960),40960**-.5);kv=rand((8192,1280),8192**-.5)
 nw=rand((8192,));rope=rand((131072,128))
 native=old.validated_consumer();source,_,pack=old.scoped_functions()
 names=['baseline','candidate','control'];report=dict(agent='decode_research',seed=a.seed,complete=False,correct=True,active=True,
  candidate='same_shape_dflash_context_cuda_graph',baseline_sha256=old.SHA,
  scope='Full 40960→8192 BF16 context FC, native RMSNorm, 8192→1280 projection and five-layer norm/RoPE/prefix write. Original target-tap producer graph; same actual rows in every arm; candidate additionally stages dynamic positions, locations and commit lengths. Synthetic weights, SM86, no target/draft model blocks or TP.',
  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  rms_wrapper_sha256=hashlib.sha256(inspect.getsource(sgl_kernel.rmsnorm).encode()).hexdigest(),
  bench=dict(instances=1,shapes={}),details={})
 def save(): (directory/'result.json').write_text(json.dumps(report,indent=2))
 save()
 for b in a.batches:
  m=b*8;cap=32768
  snapshots=[torch.empty((m,8192),device='cuda',dtype=torch.float32).normal_(generator=rng) for _ in range(5)]
  model=old.model_stub(snapshots);batch=old.types.SimpleNamespace(forward_mode=None);embed=snapshots[0].bfloat16()
  old.bench_aux.call(source,model,batch,embed,pack);torch.cuda.synchronize()
  with torch.cuda.graph(producer:=torch.cuda.CUDAGraph()):outputs=old.bench_aux.call(source,model,batch,embed,pack)
  norm=rand((5,128));eps=torch.full((5,),1e-5,device='cuda')
  loc=torch.arange(m,device='cuda',dtype=torch.int64)+1;pos=torch.zeros_like(loc);lens=torch.full((b,),4,device='cuda',dtype=torch.int32)
  buffers={}
  for name in names:
   keys=[torch.zeros((cap+1,1,128),device='cuda',dtype=torch.bfloat16) for _ in range(5)]
   values=[torch.zeros_like(x) for x in keys]
   h=torch.empty((m,8192),device='cuda',dtype=torch.bfloat16);proj=torch.empty((m,5,256),device='cuda',dtype=torch.bfloat16)
   ll,pp,cc=(loc.clone(),pos.clone(),lens.clone()) if name=='candidate' else (loc,pos,lens)
   ns=dict(torch=torch,h=h,weight=kv,proj=proj,norm=norm,eps=eps,rope=rope,pos=pp,
    temp_k=rand((5,m,1,128)),temp_v=rand((5,m,1,128)),loc=ll,lens=cc,b=b,m=m,WIDTH=8,DIM=128,LAYERS=5,
    _fused_norm_rope_kernel_stacked=old._fused_norm_rope_kernel_stacked,
    fused_direct=old.fused_direct,set_kv_buffer_prefix_valid_tiled=old.set_kv_buffer_prefix_valid_tiled)
   exec(native,ns)
   buffers[name]=dict(pool=(keys,values,old.kv_bench.pointers(keys),old.kv_bench.pointers(values)),
    h=h,proj=proj,call=ns['call'],loc=ll,pos=pp,lens=cc)
  def consumer(name):
   st=buffers[name]
   context=F.linear(outputs[1],fc)
   sgl_kernel.rmsnorm(context,nw,1e-5,out=st['h'])
   st['call'](False,st['pool'])
  # Warm native cuBLAS, RMS and Triton before capture; no timed JIT.
  for name in names:consumer(name)
  torch.cuda.synchronize()
  with torch.cuda.graph(context_graph:=torch.cuda.CUDAGraph()):consumer('candidate')
  def launch(name):
   producer.replay()
   if name=='candidate':
    st=buffers[name];st['loc'].copy_(loc);st['pos'].copy_(pos);st['lens'].copy_(lens)
    context_graph.replay()
   else:consumer(name)
  times={name:[] for name in names};events={name:[] for name in names}
  detail=dict(order=[],wrong=0,compared_cache_values=0,device_us=events)
  report['bench']['shapes'][f'b{b}']=dict(weight=1,times=times)
  report['details'][f'b{b}']=detail
  for i in range(a.rounds):
   for x in snapshots:x.normal_(generator=rng)
   loc.copy_(torch.randperm(cap,device='cuda',generator=rng)[:m]+1)
   pos.random_(40000,100000,generator=rng);lens.random_(1,9,generator=rng)
   if i%4==0:lens.fill_(8)
   order=names if i%2==0 else names[::-1];detail['order'].append(order)
   for name in order:
    torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    wall=time.perf_counter_ns();start.record();launch(name);end.record();end.synchronize()
    times[name].append((time.perf_counter_ns()-wall)/1000);events[name].append(start.elapsed_time(end)*1000)
   for name in names[1:]:
    for key in ('h','proj'):
     if not torch.equal(buffers[name][key],buffers['baseline'][key]):
      detail['failure']=dict(round=i,arm=name,stage=key,wrong=int((buffers[name][key]!=buffers['baseline'][key]).sum()))
      report['correct']=False;save();return
   index=slice(None) if i==a.rounds-1 else loc
   for part in (0,1):
    for layer in range(5):
     reference=buffers['baseline']['pool'][part][layer][index]
     detail['compared_cache_values']+=reference.numel()
     for name in names[1:]:
      wrong=int((reference.view(torch.int16)!=buffers[name]['pool'][part][layer][index].view(torch.int16)).sum())
      detail['wrong']+=wrong
   if detail['wrong']:report['correct']=False;save();return
  detail['median_wall_us']={k:statistics.median(v) for k,v in times.items()}
  detail['median_device_us']={k:statistics.median(v) for k,v in events.items()}
  print('batch',b,detail['median_wall_us'],detail['median_device_us'],flush=True);save()
  del buffers,context_graph,producer,outputs,model,snapshots,keys,values,ns,reference
  torch.cuda.synchronize();torch.cuda.empty_cache()
 report['complete']=True;save()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
 p.add_argument('--batches',nargs='+',type=int,default=[1,4,16,32]);p.add_argument('--rounds',type=int,default=12);a=p.parse_args()
 assert a.rounds%2==0
 directory=HERE/f'dflash_graph_{a.seed}';directory.mkdir(exist_ok=False)
 with (ROOT/'research_2026-11-18_rl/.gpu.lock').open('a+') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX)
  with telemetry(directory),torch.inference_mode():run(a,directory)
