"""Isolated scored-helper large-prefill profile. No source modifications or submissions."""
import argparse,ast,gc,hashlib,importlib.util,json,os,pathlib,random,statistics,sys,time
P=pathlib.Path(__file__).resolve().parent
ROOT=P.parents[3]
EXPORT=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources'

def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--out',required=True);pa.add_argument('--tokens',nargs='+',type=int,default=[753,3012,8192,16384]);pa.add_argument('--patterns',nargs='+',default=['uniform','mixed']);pa.add_argument('--samples',type=int,default=8);a=pa.parse_args();out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
 assert all(m>512 for m in a.tokens),'This profile deliberately targets scored large-M unsplit path'
 for path in (ROOT/'swarm_2026-11-16/moe/humming_vendor',ROOT/'swarm_2026-11-16/moe/phase2',ROOT/'optimization_marlin_tiles_2026-11-15'):sys.path.insert(0,str(path))
 from kernel_runtime import bootstrap
 bootstrap(ROOT/'sglang')
 import torch
 from humming.layer import HummingMethod
 from fixture_tuning_cache import cached_threshold_for_preparation
 import sglang.srt.layers.moe.fused_moe_triton as align_module
 import sglang.kernels.ops.activation.activation as act_module
 scale=load(EXPORT/'python/sglang/kernels/ops/moe/muse_spark_humming_scale.py','sglang.kernels.ops.moe.muse_spark_humming_scale')
 split=load(EXPORT/'python/sglang/kernels/ops/moe/muse_spark_humming_split.py','sglang.kernels.ops.moe.muse_spark_humming_split')
 final=load(EXPORT/'python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py','round3_scored_finalizer')
 src=EXPORT/'python/sglang/srt/layers/muse_spark/muse_spark_nvfp4_moe.py';tree=ast.parse(src.read_text());names=('_prepare_humming_experts','_humming_expert_outputs_default','_humming_expert_outputs');nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),ns)
 testsrc=ROOT/'sglang/python/sglang/test/test_marlin_utils.py';tree=ast.parse(testsrc.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_nvfp4_weight_and_ref');gen={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(testsrc),'exec'),gen)
 report={'status':'PREPARING','scope':'LocalSM86 scored large-prefill helper stage profile. No H200 or serving projection. External CUDA-event nodes instrument a second graph; clean graph measured separately.','source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [pathlib.Path(__file__),src,testsrc]},'cases':[]}
 def save():
  tmp=out/'results.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out/'results.json')
 save();torch.manual_seed(771);h=4096;e=32;raw=torch.nn.Module()
 for prefix,n in [('w13',2*h),('w2',h)]:
  weights=[];scales=[];globals_=[]
  for expert in range(e):
   w,s,g,ref=gen[node.name](n,h,torch.bfloat16);weights.append(w);scales.append(s);globals_.append(g);del ref
  for suffix,values in [('weight',weights),('weight_scale',scales),('weight_scale_2',globals_)]:setattr(raw,prefix+'_'+suffix,torch.nn.Parameter(torch.stack(values),requires_grad=False))
  del weights,scales,globals_,w,s,g,values
 for stage in ('w13','w2'):getattr(raw,stage+'_weight_scale_2').mul_(torch.linspace(.25,2.,e,device='cuda',dtype=torch.bfloat16))
 with cached_threshold_for_preparation(out):layer=ns['_prepare_humming_experts'](raw,torch)
 del raw;gc.collect();torch.cuda.empty_cache()
 report.update(status='RUNNING',device=torch.cuda.get_device_name(),SM=torch.cuda.get_device_capability(),weight_allocated_bytes=torch.cuda.memory_allocated());save()
 norm=torch.linspace(.5,1.5,h,device='cuda',dtype=torch.bfloat16);flush=torch.zeros(64*1024**2//4,device='cuda',dtype=torch.int32)
 original={'forward':HummingMethod.forward_layer,'align':align_module.moe_align_block_size,'scale':scale.scale_active_rows,'silu':act_module.silu_and_mul}
 trace=None
 def measure(label,fn,*args,**kwargs):
  if trace is None:return fn(*args,**kwargs)
  i=len(trace['entries']);before,after=trace['events'][i];before.record();value=fn(*args,**kwargs);after.record();trace['entries'].append((label,before,after));return value
 def forward(*args,**kwargs):return measure(kwargs.get('sublayer_name','gemm'),original['forward'],*args,**kwargs)
 def align(*args,**kwargs):return measure('alignment',original['align'],*args,**kwargs)
 def scales(*args,**kwargs):return measure('global_scale',original['scale'],*args,**kwargs)
 def silu(*args,**kwargs):return measure('silu',original['silu'],*args,**kwargs)
 HummingMethod.forward_layer=forward;align_module.moe_align_block_size=align;scale.scale_active_rows=scales;act_module.silu_and_mul=silu
 def cleanup():gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
 for m in a.tokens:
  for pattern in a.patterns:
   case={'m':m,'pattern':pattern,'measurements':{},'comparisons':[]};report['cases'].append(case);save()
   x=torch.empty((m,h),device='cuda',dtype=torch.bfloat16);ids=torch.empty((m,8),device='cuda',dtype=torch.int32);scores=torch.empty((m,8),device='cuda')
   weights=[24. if v<8 else 1. for v in range(256)]
   def set_inputs(generation):
    rng=random.Random(1641+m+generation);rows=[]
    for token in range(m):
     if pattern=='uniform':row=rng.sample(range(256),8)
     elif pattern=='mixed':
      row=[]
      while len(row)<8:
       v=rng.choices(range(256),weights=weights,k=1)[0]
       if v not in row:row.append(v)
     else:raise ValueError(pattern)
     assert len(set(row))==8
     rows.append([v if v<e else -1 for v in row])
    ids.copy_(torch.tensor(rows,device='cuda',dtype=torch.int32));torch.manual_seed(2341+m+generation);x.copy_(torch.randn_like(x)/20);scores.copy_(torch.rand_like(scores));scores.div_(scores.sum(-1,keepdim=True))
    return [sum(v==expert for row in rows for v in row) for expert in range(e)]
   case['expert_counts']=set_inputs(0);case['selected_configs']={s:next(c for lo,hi,c in intervals if lo<m*8<=hi) for s,intervals in layer.fp32_configs.items()}
   def call():
    value,local=ns['_humming_expert_outputs'](layer,x,ids,torch,defer_downscale=True,global_experts=256)
    return measure('finalizer',final.fused_post_expert_norm_weighted_sum,value,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=local,expert_output_scale=layer.w2_post_scale)
   references=[]
   for kind in ('clean','profiled'):
    trace=None
    for _ in range(2):tmp=call();del tmp
    cleanup()
    if kind=='profiled':
     eventpairs=[(torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)) for _ in range(12)]
     for b,c in eventpairs:b.record();c.record()
     torch.cuda.synchronize();trace={'events':eventpairs,'entries':[]}
    torch.cuda.reset_peak_memory_stats();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):result=call()
    entries=list(trace['entries']) if trace else [];trace=None
    for generation in range(2):
     set_inputs(generation);graph.replay();torch.cuda.synchronize();cpu=result.cpu()
     assert torch.isfinite(cpu).all()
     if kind=='clean':references.append(cpu)
     else:
      equal=torch.equal(cpu.view(torch.int16),references[generation].view(torch.int16));case['comparisons'].append({'generation':generation,'finite':True,'bitwise':equal});assert equal
    set_inputs(0)
    rec={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'event_labels':[label for label,_,_ in entries],'samples':[]};case['measurements'][kind]=rec;save()
    for cold in (False,True):
     until=time.monotonic()+.3
     while time.monotonic()<until:graph.replay();torch.cuda.synchronize()
     for sample_id in range(a.samples):
      if cold:flush.add_(1)
      begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record();graph.replay();end.record();end.synchronize()
      phases=[{'label':label,'us':b.elapsed_time(c)*1000} for label,b,c in entries]
      rec['samples'].append({'cold':cold,'sample_id':sample_id,'total_us':begin.elapsed_time(end)*1000,'phases':phases})
     save()
    del graph,result,cpu,entries
    if kind=='profiled':del eventpairs
    cleanup()
   for cold in (False,True):
    med={kind:statistics.median(v['total_us'] for v in case['measurements'][kind]['samples'] if v['cold']==cold) for kind in ('clean','profiled')}
    case.setdefault('instrumentation_ratio',[]).append({'cold':cold,'profiled_over_clean':med['profiled']/med['clean'],'median_us':med,'warning':'Sequential graphs; diagnostics only, not paired candidate speedup.'})
   case['status']='COMPLETE';save();print(json.dumps({'m':m,'pattern':pattern,'instrumentation':case['instrumentation_ratio']}),flush=True)
   del references,x,ids,scores;cleanup()
 report['status']='COMPLETE_LOCAL_BASELINE_PROFILE';save()
if __name__=='__main__':main()
