"""Replicated DFlash target-feature FC sharding study, local component only."""
from __future__ import annotations
import argparse,gc,hashlib,json,pathlib,statistics,time
ROOT=pathlib.Path(__file__).resolve().parents[3];OUT=pathlib.Path(__file__).resolve().parent

def main():
 import torch
 import torch.nn.functional as F
 parser=argparse.ArgumentParser();parser.add_argument('--rows',nargs='+',type=int,default=[1,8,32,128,256,512,768,1024,4096]);parser.add_argument('--out',required=True);parser.add_argument('--samples',type=int,default=12);parser.add_argument('--force-full-reduction',action='store_true');parser.add_argument('--correctness-only',action='store_true');args=parser.parse_args()
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 inherited_reduction_flag=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
 if args.force_full_reduction:torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 torch.manual_seed(617)
 assert torch.cuda.is_available()
 free,total=torch.cuda.mem_get_info();required=(40960*8192+max(args.rows)*40960+4*max(args.rows)*8192)*2+512*1024**2
 if free<required:raise RuntimeError(f'Capacity preflight:free={free},required={required}')
 weight=torch.randn((8192,40960),dtype=torch.bfloat16,device='cuda')*.003
 norm_weight=torch.ones(8192,dtype=torch.bfloat16,device='cuda')
 flush=torch.empty(64*1024**2//4,device='cuda',dtype=torch.float32)
 def metrics(a,b):
  af=a.float();bf=b.float();difference=af-bf
  return {'bitwise':torch.equal(a.view(torch.uint8),b.view(torch.uint8)),'different_elements':int((a!=b).sum()),'elements':a.numel(),'max_abs':float(difference.abs().max()),'relative_l2':float(torch.linalg.vector_norm(difference)/torch.linalg.vector_norm(af).clamp_min(1e-30)),'finite':bool(torch.isfinite(b).all())}
 def norm(x):return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)*norm_weight).to(x.dtype)
 def capture(call):
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):call()
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):out=call()
  return graph,out
 def timed(graph,cold):
  if cold:flush.add_(1)
  start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
  start.record();graph.replay();end.record();end.synchronize();return start.elapsed_time(end)*1000
 rows=[]
 for m in args.rows:
  torch.cuda.reset_peak_memory_stats()
  x=torch.randn((m,40960),device='cuda',dtype=torch.bfloat16)
  numerical=[]
  for generation in range(3):
   if generation:x.mul_(-.5).add_(.03125)
   base=F.linear(x,weight)
   column=torch.cat([F.linear(x,weight[r*1024:(r+1)*1024]) for r in range(8)],dim=-1)
   nr=(m+7)//8
   row=torch.cat([F.linear(x[r*nr:min((r+1)*nr,m)],weight) for r in range(8) if r*nr<m],dim=0)
   numerical.append({'generation':generation,'column':metrics(base,column),'row':metrics(base,row),'column_norm':metrics(norm(base),norm(column)),'row_norm':metrics(norm(base),norm(row))})
   del base,column,row
  if args.correctness_only:
   rows.append({'rows':m,'numerical':numerical})
   print(json.dumps({'rows':m,'column_bitwise':all(g['column']['bitwise'] for g in numerical),'row_bitwise':all(g['row']['bitwise'] for g in numerical)}),flush=True)
   del x
   gc.collect();torch.cuda.empty_cache()
   continue
  b,bo=capture(lambda:F.linear(x,weight))
  c,co=capture(lambda:F.linear(x,weight[:1024]))
  r,ro=capture(lambda:F.linear(x[:(m+7)//8],weight))
  graphs={'base':b,'column_rank':c,'row_rank':r}
  changed=[]
  for generation in range(3):
   x.mul_(-.5).add_(.03125)
   for g in graphs.values():g.replay()
   eager=F.linear(x,weight)
   changed.append({'generation':generation,'baseline_graph':metrics(eager,bo),'column_rank_graph':metrics(F.linear(x,weight[:1024]),co),'row_rank_graph':metrics(F.linear(x[:(m+7)//8],weight),ro)})
   del eager
  timings={}
  for cold in (False,True):
   samples={key:[] for key in ('A1','Bcol','A2','Brow')}
   for repetition in range(args.samples):
    order=[('A1',b),('Bcol',c),('A2',b),('Brow',r)]
    if repetition%2:order.reverse()
    for name,g in order:samples[name].append(timed(g,cold))
   med={key:statistics.median(value) for key,value in samples.items()}
   baseline=statistics.median(samples['A1']+samples['A2'])
   ratio=med['A1']/med['A2']
   timings['cold64MiB' if cold else 'warm']={'samples_us':samples,'medians_us':med,'aa_ratio':ratio,'aa_pass':.95<=ratio<=1.05,'local_baseline_over_column_rank':baseline/med['Bcol'],'local_baseline_over_row_rank':baseline/med['Brow'],'network_cost_included':False}
  row_result={'rows':m,'numerical':numerical,'changed_graph':changed,'timings':timings,'memory':{'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}}
  rows.append(row_result)
  print(json.dumps({'rows':m,'column_bitwise':all(i['column']['bitwise'] for i in numerical),'row_bitwise':all(i['row']['bitwise'] for i in numerical),'warm':timings['warm']['medians_us'],'cold':timings['cold64MiB']['medians_us']}),flush=True)
  del x,b,c,r,bo,co,ro,graphs
  gc.collect();torch.cuda.empty_cache()
 report={'status':'LOCAL_SM86_COMPONENT_STUDY_NO_DISTRIBUTED_OR_SERVING_VALIDATION','torch':torch.__version__,'device':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'shape':{'input_features':40960,'output_features':8192,'tp':8,'weight_bytes':weight.numel()*weight.element_size(),'provenance':'RadixArk/Muse-Spark-Open-DFlash2 publicconfig hidden8192 times5targettaps'},'precision':{'input_weight':'BF16','allow_tf32':False,'inherited_bf16_reduction_flag':inherited_reduction_flag,'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction},'rows':rows,'limitations':['Synthetic weights and activations','No collective/parallel-rank cost in timing; reported ratios are local GEMM-only','Actual serving precision flags need comparison; local inherited and selected reduction flags are recorded','Norm oracle torch FP32 RMSnorm, not actual SGLang norm kernel','RTX3050 is notH200; no throughput frontier prediction']}
 pathlib.Path(args.out).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()


