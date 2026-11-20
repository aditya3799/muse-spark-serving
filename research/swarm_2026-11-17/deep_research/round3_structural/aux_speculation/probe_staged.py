"""Bounded exact-operand / same-precision algorithm study, not serving integration."""
import argparse,ast,gc,hashlib,importlib.util,json,os,random,statistics,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[3];OLD=ROOT/'swarm_2026-11-16/moe';EXPORT=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--experts',type=int,nargs='+',default=[1,2]);ap.add_argument('--rows',type=int,nargs='+',default=[32,128,512,2048]);ap.add_argument('--samples',type=int,default=8);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 for f in (OLD/'humming_vendor',OLD/'phase2',ROOT/'optimization_marlin_tiles_2026-11-15'):sys.path.insert(0,str(f))
 from kernel_runtime import bootstrap
 bootstrap(ROOT/'sglang')
 import torch
 from fixture_tuning_cache import cached_threshold_for_preparation
 from staged_ops import dequant,scatter
 import sglang.kernels.ops.activation.activation as actmod
 native_act=actmod.silu_and_mul
 torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
 def load(name,path):
  sp=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(sp);sys.modules[name]=mod;sp.loader.exec_module(mod);return mod
 sc=load('sglang.kernels.ops.moe.muse_spark_humming_scale',EXPORT/'python/sglang/kernels/ops/moe/muse_spark_humming_scale.py')
 split=load('sglang.kernels.ops.moe.muse_spark_humming_split',EXPORT/'python/sglang/kernels/ops/moe/muse_spark_humming_split.py')
 finalmod=load('staged_scored_finalizer',EXPORT/'python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py')
 src=EXPORT/'python/sglang/srt/layers/muse_spark/muse_spark_nvfp4_moe.py';tree=ast.parse(src.read_text());names=['_prepare_humming_experts','_humming_expert_outputs_default','_humming_expert_outputs'];nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),ns)
 stock=ROOT/'sglang/python/sglang/test/test_marlin_utils.py';tree=ast.parse(stock.read_text());gen=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_nvfp4_weight_and_ref');gs={'torch':torch};exec(compile(ast.Module(body=[gen],type_ignores=[]),str(stock),'exec'),gs)
 report={'status':'RUNNING','scope':'Local E1/E2 H4096 I4096 per-expert chain with full scored finalizer. Each selected expert processes every input row; changed slot permutation and masked routes are correctness-only. Not generic E32 dispatch or serving proof.','operand_contract':'Exact finite NVFP4 group-scaled BF16 values; global BF16 scales remain after GEMMs.','algorithm_contract':'cuBLAS BF16 inputs/outputs, FP32 accumulation; BF16 reduced-precision reduction disabled. Changed accumulation order is held, not called lossless serving proof.','comparisons':[],'timings':[],'workspace_bytes':64*1024**2,'checks':[],'timing_environment':'External user-authorized VoxCPM TTS remains running; all timing is provisional even when pair controls pass.','memory_preflight':[]}
 def save():(args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
 def compare(a,b,**meta):
  a=a.contiguous().view(-1);b=b.contiguous().view(-1);finite=True;bits=True;max_abs=0.;sum_sq=0.;max_rel=0.
  for lo in range(0,a.numel(),1<<20):
   aa=a[lo:lo+(1<<20)];bb=b[lo:lo+(1<<20)];finite=finite and bool(torch.isfinite(aa).all() and torch.isfinite(bb).all());bits=bits and torch.equal(aa.view(torch.int16),bb.view(torch.int16))
   err=(aa.float()-bb.float()).abs();max_abs=max(max_abs,float(err.max()));sum_sq+=float(err.square().sum());max_rel=max(max_rel,float((err/bb.float().abs().clamp_min(1e-6)).max()))
  report['comparisons'].append(dict(meta,finite=finite,bitwise=bits,max_abs=max_abs,rms_abs=(sum_sq/max(a.numel(),1))**.5,max_relative_to_1e_minus6_floor=max_rel));save();assert finite,meta

 free,total=torch.cuda.mem_get_info();report['initial_memory']={'free_bytes':free,'total_bytes':total,'fixture_budget':384*1024**2,'reserve':512*1024**2};save()
 if free<(384+512)*1024**2:
  report['status']='HELD_INITIAL_MEMORY_RESERVE';save();return
 flush=torch.empty(64*1024**2,device='cuda',dtype=torch.uint8);h=4096;topk=8;norm=torch.linspace(.5,1.5,h,device='cuda',dtype=torch.bfloat16)
 weight_workspace=torch.empty((2*h,h),device='cuda',dtype=torch.bfloat16)
 def capture(fn):
  for _ in range(2):fn()
  torch.cuda.synchronize();observed.clear();gc.collect();torch.cuda.empty_cache();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):value=fn()
  return g,value
 def sample(graph,cold):
  if cold:flush.fill_(89)
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();graph.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
 for e in args.experts:
  torch.manual_seed(799+e);raw=torch.nn.Module()
  for stage,n in (('w13',2*h),('w2',h)):
   ws=[];ss=[];gg=[]
   for expert in range(e):
    w,s,g,ref=gs[gen.name](n,h,torch.bfloat16);ws.append(w);ss.append(s);gg.append(g);del ref
   for suffix,values in (('weight',ws),('weight_scale',ss),('weight_scale_2',gg)):setattr(raw,stage+'_'+suffix,torch.nn.Parameter(torch.stack(values),requires_grad=False))
   del ws,ss,gg,w,s,g,values
  with cached_threshold_for_preparation(args.out):baseline=ns['_prepare_humming_experts'](raw,torch)
  print('prepared_E%d'%e,flush=True)
  # Native GPU dequant against independent unpack reference, excluding global scale.
  for stage,n in (('w13',2*h),('w2',h)):
   w=getattr(raw,stage+'_weight')[0];sf=getattr(raw,stage+'_weight_scale')[0];out=weight_workspace[:n]
   dequant(w,sf,out)
   lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],device='cuda',dtype=torch.float32)
   # Bounded independent sample covers many groups and both nibbles.
   idx=torch.arange(65536,device='cuda',dtype=torch.int64)*((n*h)//65536);code=(w.flatten()[idx//2].long()>>((idx%2)*4))&15
   ref=(lut[code]*sf.flatten().float()[idx//16]).bfloat16();compare(out.flatten()[idx],ref,e=e,stage=stage+'_dequant_sample');assert torch.equal(out.flatten()[idx].view(torch.int16),ref.view(torch.int16))
   del lut,idx,code,ref,w,sf,out
  for rows in args.rows:
   gc.collect();torch.cuda.empty_cache();free,total=torch.cuda.mem_get_info();T=rows*h*2;S=rows*topk*h*2
   estimated_additional=6*S+8*T+2*e*T+96*1024**2;reserve=512*1024**2
   report['memory_preflight'].append({'experts':e,'rows':rows,'free_bytes':free,'total_bytes':total,'estimated_additional_peak':estimated_additional,'required_reserve':reserve,'fits':free>=estimated_additional+reserve});save()
   if free<estimated_additional+reserve:
    report.setdefault('skipped',[]).append({'experts':e,'rows':rows,'reason':'External workload plus512MiB reserve leaves insufficient estimated memory'});save();continue
   x=torch.empty((rows,h),device='cuda',dtype=torch.bfloat16);ids=torch.empty((rows,topk),device='cuda',dtype=torch.int32);scores=torch.empty((rows,topk),device='cuda')
   gateup=torch.empty((rows,2*h),device='cuda',dtype=torch.bfloat16);activation=torch.empty((rows,h),device='cuda',dtype=torch.bfloat16);tempdown=torch.empty_like(activation)
   scale_ids=[torch.full((rows,),expert,device='cuda',dtype=torch.int32) for expert in range(e)]
   def inputs(generation):
    torch.manual_seed(291+generation);x.copy_(torch.randn_like(x)/20);scores.copy_(torch.rand_like(scores));scores.div_(scores.sum(-1,keepdim=True));ids.fill_(-1)
    for expert in range(e):ids[:,(expert+generation)%topk]=expert
    if generation==2:ids[::3,:]=-1
   def finish(down):return finalmod.fused_post_expert_norm_weighted_sum(down,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=ids,expert_output_scale=baseline.w2_post_scale)
   observed=[]
   def observer(x,out,*a,**kw):
    result=native_act(x,out,*a,**kw);observed[:]=[out];return result
   def original():
    actmod.silu_and_mul=observer
    try:down,local=ns['_humming_expert_outputs'](baseline,x,ids,torch,defer_downscale=True,global_experts=256)
    finally:actmod.silu_and_mul=native_act
    return down,observed[-1],finish(down)
   def staged(debug=False):
    down=torch.empty((rows,topk,h),device='cuda',dtype=torch.bfloat16);actout=torch.empty_like(down) if debug else None
    for expert in range(e):
     dequant(raw.w13_weight[expert],raw.w13_weight_scale[expert],weight_workspace)
     torch.mm(x,weight_workspace.t(),out=gateup)
     sc.scale_active_rows(gateup,scale_ids[expert],baseline.w13_post_scale)
     native_act(gateup,activation,expert_ids=scale_ids[expert],expert_step=1)
     if debug:scatter(activation,ids,actout,expert)
     dequant(raw.w2_weight[expert],raw.w2_weight_scale[expert],weight_workspace[:h])
     torch.mm(activation,weight_workspace[:h].t(),out=tempdown)
     scatter(tempdown,ids,down,expert)
    return down,actout,finish(down)
   inputs(0);ga,oa=capture(original);gb,ob=capture(staged);gd,od=capture(lambda:staged(True))
   for generation in range(3):
    if generation==1:
     # Mutate canonical bytes, then repack only the reference outside all timed calls.
     raw.w13_weight.bitwise_xor_(1);raw.w2_weight.bitwise_xor_(16)
     fresh=ns['_prepare_humming_experts'](raw,torch)
     assert fresh.fp32_configs==baseline.fp32_configs
     for stage in ('w13','w2'):
      for suffix in ('weight','weight_scale','post_scale'):
       getattr(baseline,stage+'_'+suffix).copy_(getattr(fresh,stage+'_'+suffix))
     del fresh
    inputs(generation);ga.replay();gb.replay();gd.replay();torch.cuda.synchronize();active=ids>=0
    compare(od[1][active],oa[1][active.flatten()],e=e,rows=rows,generation=generation,stage='activation')
    compare(ob[0][active],oa[0][active],e=e,rows=rows,generation=generation,stage='down')
    compare(ob[2],oa[2],e=e,rows=rows,generation=generation,stage='final')
    compare(ob[2],od[2],e=e,rows=rows,generation=generation,stage='debug_fast_control');assert torch.equal(ob[2].view(torch.int16),od[2].view(torch.int16))
   inputs(0);ga.replay();torch.cuda.synchronize();aa=oa[2].clone();ga.replay();torch.cuda.synchronize()
   compare(oa[2],aa,e=e,rows=rows,stage='baseline_repeat_control');assert torch.equal(oa[2].view(torch.int16),aa.view(torch.int16));del aa
   for cold in (False,True):
    start=time.monotonic()
    while time.monotonic()-start<.3:
     for graph in (ga,gb):sample(graph,cold)
    samples={key:[] for key in ('A1','B1','A2','B2')};rng=random.Random(931+rows)
    for repeat in range(args.samples):
     arms=list(samples);rng.shuffle(arms)
     for arm in arms:samples[arm].append(sample(ga if arm.startswith('A') else gb,cold))
    med={k:statistics.median(v) for k,v in samples.items()};ratio=lambda a,b:max(a,b)/min(a,b);half=args.samples//2
    controls={'AA':ratio(med['A1'],med['A2']),'BB':ratio(med['B1'],med['B2'])};drift={k:ratio(statistics.median(v[:half]),statistics.median(v[half:])) for k,v in samples.items()}
    report['timings'].append({'experts':e,'rows_per_expert':rows,'cold_sweep':cold,'samples_us':samples,'median_us':med,'controls':controls,'half_drift':drift,'qualified':max(list(controls.values())+list(drift.values()))<=1.1,'speedup':statistics.median(samples['A1']+samples['A2'])/statistics.median(samples['B1']+samples['B2'])});save()
   print(json.dumps({'e':e,'rows':rows,'timings':report['timings'][-2:]}),flush=True)
   observed.clear();del ga,gb,gd,oa,ob,od,x,ids,scores,gateup,activation,tempdown,scale_ids;gc.collect();torch.cuda.empty_cache()
  del raw,baseline;torch.cuda.empty_cache()
 report.update(status='EXPERIMENT_COMPLETE_NOT_SERVING_QUALIFIED',strict_bitwise_gate=all(row['bitwise'] for row in report['comparisons']),peak_allocated=torch.cuda.max_memory_allocated(),device=torch.cuda.get_device_name());save()
if __name__=='__main__':main()
