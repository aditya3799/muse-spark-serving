"""Complete scored-helper comparison, default CPU preflight; no guard bypass.
Requires reviewed WK32/N4096 admission and an exclusive root hardware lease.
"""
from pathlib import Path
import argparse,ast,copy,gc,hashlib,importlib.util,json,random,statistics,sys,time
from swap_full_helper_plan import manifest,plan,P,ROOT,HOPPER,CONFIG
OLD=ROOT/'swarm_2026-11-16/moe';EXPORT=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--gpu',action='store_true');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 assert not(a.out/'results.json').exists(),'Preserve prior attempts'
 report=manifest();report.update(checks=[],timings=[],memory=[],kernels=[],route_counts=[])
 def save():(a.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
 save()
 if not a.gpu or any(x['status']!='ADMITTED' for x in report['admission']):
  print(json.dumps({'status':report['status'],'gpu_started':False}));return
 # Reviewed v3 admitsWK32/N4096 after independent micro/pipeline gates; this fullhelper remains a new numerical research gate and requires root lease.
 for d in (OLD/'humming_vendor',OLD/'phase2',ROOT/'optimization_marlin_tiles_2026-11-15',HOPPER,P.parent/'aux_speculation'):sys.path.insert(0,str(d))
 from kernel_runtime import bootstrap
 bootstrap(ROOT/'sglang')
 import torch
 from humming.layer import HummingMethod
 from humming import ops
 from swap_ab_humming_runtime_v3 import SwapHummingKernel
 from fixture_tuning_cache import cached_threshold_for_preparation
 from adversarial_fixture import weight_fixture,activation_fixture
 import sglang.kernels.ops.activation.activation as actmod
 assert torch.cuda.get_device_capability()==(8,6) and torch.cuda.get_device_properties().multi_processor_count==20
 reserve=512*1024**2;budget=int(1.15*1024**3);entry_alloc=torch.cuda.memory_allocated();entry_reserved=torch.cuda.memory_reserved();torch.cuda.reset_peak_memory_stats()
 def memory(label,extra=0):
  free,total=torch.cuda.mem_get_info();owned=torch.cuda.memory_reserved()-entry_reserved;peak_owned=torch.cuda.max_memory_allocated()-entry_alloc
  row={'label':label,'free':free,'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'own_reserved_delta':owned,'own_peak_allocated_delta':peak_owned,'peak_allocated':torch.cuda.max_memory_allocated(),'next_estimated':extra};report['memory'].append(row);save()
  assert free>=extra+reserve and owned+extra<=budget and peak_owned<=budget,('Memory budget',row)
 memory('initial',700*1024**2)
 def load(name,path):
  spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
 scale=load('sglang.kernels.ops.moe.muse_spark_humming_scale',EXPORT/'kernels/ops/moe/muse_spark_humming_scale.py')
 split=load('sglang.kernels.ops.moe.muse_spark_humming_split',EXPORT/'kernels/ops/moe/muse_spark_humming_split.py')
 final=load('fullhelper_scored_finalizer',EXPORT/'kernels/ops/moe/muse_spark_post_expert_finalize_triton.py')
 src=EXPORT/'srt/layers/muse_spark/muse_spark_nvfp4_moe.py';tree=ast.parse(src.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in('_prepare_humming_experts','_humming_expert_outputs_default','_humming_expert_outputs')];ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),ns)
 files=[Path(__file__),P/'swap_full_helper_plan.py',src,Path(split.__file__),Path(scale.__file__),Path(final.__file__),HOPPER/'swap_ab_humming_v3.cuh',HOPPER/'swap_ab_humming_codegen_v3.py',HOPPER/'swap_ab_humming_runtime_v3.py',P.parent/'aux_speculation/adversarial_fixture.py',CONFIG,P/'SWAP_FULL_HELPER_PREREQUISITES.json',*sorted((OLD/'humming_vendor/humming').rglob('*.py')),*sorted((OLD/'humming_vendor/humming/include').rglob('*.cuh'))]
 extra_sources=[ROOT/'optimization_marlin_tiles_2026-11-15/kernel_runtime.py',OLD/'phase2/fixture_tuning_cache.py',Path(actmod.__file__),ROOT/'sglang/python/sglang/kernels/jit/csrc/elementwise/activation.cuh']
 files+=extra_sources
 depdir=a.out/'dependency_snapshots';depdir.mkdir(exist_ok=True)
 for i,f in enumerate(extra_sources):(depdir/('%02d_'%i+f.name)).write_bytes(f.read_bytes())
 report['source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in files};report['launch_policy']={'num_sms':20,'num_ctas_per_sm':1,'preserved_all_arms':True,'warning':'OnepersistentCTA/SM; resourcecapacity alone is not a measured residencygain.'};save()
 e=8;m=32;h=4096;topk=8;raw=torch.nn.Module()
 for stage,n in [('w13',8192),('w2',4096)]:
  # Preallocate rawweights, avoiding stack of duplicateexpert tensors.
  w=torch.empty((e,n,h//2),device='cuda',dtype=torch.uint8);s=torch.empty((e,n,h//16),device='cuda',dtype=torch.float8_e4m3fn);g=torch.empty(e,device='cuda',dtype=torch.bfloat16)
  for expert in range(e):
   memory('before_fixture_%s_%d'%(stage,expert),200*1024**2)
   ww,ss,gg=weight_fixture(n,h,stage,expert,0,device='cuda');w[expert].copy_(ww);s[expert].copy_(ss);g[expert].copy_(gg);del ww,ss,gg
  for suffix,t in [('weight',w),('weight_scale',s),('weight_scale_2',g)]:setattr(raw,stage+'_'+suffix,torch.nn.Parameter(t,requires_grad=False))
  del w,s,g,t
 gc.collect();torch.cuda.empty_cache();memory('before_prepare',300*1024**2)
 with cached_threshold_for_preparation(a.out):baseline=ns['_prepare_humming_experts'](raw,torch)
 del raw;gc.collect();torch.cuda.empty_cache();baseline.fp32_configs=copy.deepcopy(json.loads(CONFIG.read_text())['indexed'])
 actual=split.choose_partition_configs(baseline,256,256);assert actual['cutoff']==16
 for name in ('light','heavy'):
  for stage in ('w13','w2'):
   assert actual[name][stage]['block_shape']==plan('reference')[name][stage]['block_shape'] and actual[name][stage]['warp_shape']==plan('reference')[name][stage]['warp_shape']
 def transpose_copy(source,dest):
  sf=source.flatten();df=dest.flatten()
  for lo in range(0,sf.numel(),1<<20):
   w=sf[lo:lo+(1<<20)].to(torch.int64)&0xffffffff
   df[lo:lo+(1<<20)].copy_(((w&0xf00ff00f)|((w&0x00f000f0)<<4)|((w&0x0f000f00)>>4)).to(torch.int32))
 memory('before_canonical_copy',256*1024**2)
 canonical={stage:torch.empty_like(getattr(baseline,stage+'_weight'))for stage in ('w13','w2')}
 for stage in canonical:transpose_copy(getattr(baseline,stage+'_weight'),canonical[stage])
 layers={'A':baseline}
 for arm in ('B','C'):
  layer=copy.copy(baseline);layer._parameters=dict(baseline._parameters);layer._swap_arm=arm
  for stage in canonical:setattr(layer,stage+'_weight',torch.nn.Parameter(canonical[stage],requires_grad=False))
  layers[arm]=layer
 assert all(getattr(layers['B'],s+'_weight').data_ptr()==getattr(layers['C'],s+'_weight').data_ptr()!=getattr(baseline,s+'_weight').data_ptr()for s in canonical)
 weight_pointers={s:(getattr(baseline,s+'_weight').data_ptr(),getattr(baseline,s+'_weight_scale').data_ptr(),canonical[s].data_ptr())for s in canonical}
 memory('resident_weights',150*1024**2)
 tables={};kept=[];armplan={'B':plan('swap_same'),'C':plan('swap_tiles')}
 for arm in ('B','C'):
  for part in ('light','heavy'):
   for stage,c in armplan[arm][part].items():
    key=(stage,json.dumps(c,sort_keys=True))
    if key not in tables:
     ker=SwapHummingKernel(**c,research_direct=True,research_semistatic=True);kept.append(ker);tables[key]=torch.tensor([0,1<<30,ker.kernel_id,20],dtype=torch.int64)
     report['kernels'].append({'arm':arm,'partition':part,'stage':stage,'config':c,'kernel_id':ker.kernel_id,'shared_bytes':ker.estimated_smem_size,'cubin_sha256':hashlib.sha256(Path(ker.kernel_filename).read_bytes()).hexdigest()});save()
 original_forward=HummingMethod.forward_layer
 def dispatch(layer,**kw):
  if not hasattr(layer,'_swap_arm'):return original_forward(layer,**kw)
  c=kw['tuning_config'];stage=kw['sublayer_name'];key=(stage,json.dumps(c,sort_keys=True))
  return ops.launch_kernel(configs=tables[key],inputs=kw['inputs'],outputs=kw['outputs'],weight=getattr(layer,stage+'_weight'),weight_scale=getattr(layer,stage+'_weight_scale'),sorted_ids=kw['sorted_ids'],expert_ids=kw['expert_ids'],num_tokens_padded=kw['num_tokens_padded'],locks=layer.locks,top_k=kw['top_k'],valid_shape_m=kw['valid_shape_m'])
 HummingMethod.forward_layer=staticmethod(dispatch)
 stree=ast.parse(Path(split.__file__).read_text());snode=next(n for n in stree.body if isinstance(n,ast.FunctionDef)and n.name=='split_expert_outputs')
 def choose(layer,routes,global_e):
  assert routes==256 and global_e==256 and layer._swap_arm in armplan
  return armplan[layer._swap_arm]
 sn={'torch':torch,'partition_routes':split.partition_routes,'choose_partition_configs':choose};exec(compile(ast.Module(body=[snode],type_ignores=[]),str(split.__file__),'exec'),sn)
 def forbidden_fallback(*args,**kwargs):raise AssertionError('Canonicalfallbacknotqualified')
 x=torch.empty(m,h,device='cuda',dtype=torch.bfloat16);ids=torch.empty(m,topk,device='cuda',dtype=torch.int32);scores=torch.empty(m,topk,device='cuda');norm=torch.linspace(.5,1.5,h,device='cuda',dtype=torch.bfloat16);flush=torch.empty(64*1024**2,device='cuda',dtype=torch.uint8)
 def inputs(pattern,generation):
  rng=random.Random(997+generation);rows=[]
  for token in range(m):
   if pattern=='uniform':chosen=rng.sample(range(256),8)
   elif pattern=='hot':chosen=list(range(8))
   elif pattern=='mixed':
    chosen=([0]if token<17 else [])+([7]if token<(15 if generation%2==0 else 17)else []);chosen+=rng.sample(range(8,256),8-len(chosen))
   elif pattern=='last':chosen=[7]+rng.sample(range(8,256),7)
   else:chosen=rng.sample(range(8,256),8)
   rng.shuffle(chosen);assert len(set(chosen))==8;rows.append([v if v<8 else -1 for v in chosen])
  if pattern=='tail_invalid':
   for token in range(m//2):rows[token]=list(range(8))
  ids.copy_(torch.tensor(rows,device='cuda',dtype=torch.int32));x.copy_(activation_fixture(m,h,generation,device='cuda'));torch.manual_seed(331+generation);scores.copy_(torch.rand_like(scores));scores.div_(scores.sum(-1,keepdim=True));report['route_counts'].append({'pattern':pattern,'generation':generation,'counts':[sum(row.count(i)for row in rows)for i in range(e)]});save()
 native_act=actmod.silu_and_mul;observed={};numeric=True;active_arm='A'
 def observer(value,out,*args,**kw):
  raw=value.clone();result=native_act(value,out,*args,**kw);observed[active_arm]=(raw,out);return result
 def full(arm):
  nonlocal active_arm
  active_arm=arm;layer=layers[arm]
  if numeric:actmod.silu_and_mul=observer
  try:
   if arm=='A':down,_=ns['_humming_expert_outputs'](layer,x,ids,torch,defer_downscale=True,global_experts=256)
   else:down,_=sn['split_expert_outputs'](forbidden_fallback,layer,x,ids,torch,256,defer_downscale=True)
  finally:actmod.silu_and_mul=native_act
  result=final.fused_post_expert_norm_weighted_sum(down,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=ids,expert_output_scale=baseline.w2_post_scale)
  return (result,down,*observed[arm])if numeric else result
 def capture(fn):
  for _ in range(2):fn()
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):value=fn()
  return g,value
 def compare(candidate,ref,label):
  a0=candidate.contiguous();b0=ref.contiguous();ok=bool(torch.isfinite(a0).all()and torch.isfinite(b0).all())and torch.equal(a0.view(torch.int16),b0.view(torch.int16));report['checks'].append({'label':label,'elements':a0.numel(),'finite_bitwise':ok});save();assert ok,label
 inputs('uniform',0);graphs={};outputs={}
 for arm in 'ABC':
  memory('before_numeric_capture_'+arm,40*1024**2);graphs[arm],outputs[arm]=capture(lambda arm=arm:full(arm))
 assert len({v[0].data_ptr()for v in outputs.values()})==3
 memory('numeric_graphs')
 for generation in range(3):
  if generation:
   memory('before_weight_scale_mutation',64*1024**2)
   for stage in canonical:
    w=getattr(baseline,stage+'_weight');sf=getattr(baseline,stage+'_weight_scale');wf=w.flatten()
    for lo in range(0,wf.numel(),1<<20):wf[lo:lo+(1<<20)].bitwise_xor_(0x08800880)
    sf.view(torch.uint8).copy_(((sf.view(torch.uint8).to(torch.int16)+11)%127).to(torch.uint8));transpose_copy(w,canonical[stage]);getattr(baseline,stage+'_post_scale').mul_(.75)
  assert weight_pointers=={s:(getattr(baseline,s+'_weight').data_ptr(),getattr(baseline,s+'_weight_scale').data_ptr(),canonical[s].data_ptr())for s in canonical}
  memory('after_generation_mutation')
  for pattern in ('uniform','hot','mixed','last','all_remote','tail_invalid'):
   inputs(pattern,generation)
   for v in outputs.values():v[0].fill_(37.);v[1].fill_(37.)
   for graph in graphs.values():graph.replay()
   torch.cuda.synchronize();active=ids>=0
   for arm in 'BC':
    compare(outputs[arm][0],outputs['A'][0],(arm,generation,pattern,'final'))
    if bool(active.any()):
     compare(outputs[arm][1][active],outputs['A'][1][active],(arm,generation,pattern,'down'))
     compare(outputs[arm][2][active.flatten()],outputs['A'][2][active.flatten()],(arm,generation,pattern,'scaledW13'))
     compare(outputs[arm][3][active.flatten()],outputs['A'][3][active.flatten()],(arm,generation,pattern,'activation'))
   invalid=~active.any(-1);assert all(bool((v[0][invalid]==0).all())for v in outputs.values())
 observed.clear();del graph,v,graphs,outputs;gc.collect();torch.cuda.empty_cache();numeric=False
 graphs={};outputs={};inputs('uniform',0)
 for arm in 'ABC':
  memory('before_clean_capture_'+arm,20*1024**2);graphs[arm],outputs[arm]=capture(lambda arm=arm:full(arm))
 assert len({v.data_ptr()for v in outputs.values()})==3;memory('clean_graphs')
 def sample(graph,cold):
  if cold:flush.fill_(71)
  start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();graph.replay();end.record();end.synchronize();return start.elapsed_time(end)*1000
 for pattern in ('uniform','hot','mixed','all_remote'):
  inputs(pattern,0)
  for graph in graphs.values():graph.replay()
  torch.cuda.synchronize()
  for arm in 'BC':compare(outputs[arm],outputs['A'],(arm,pattern,'cleanfinal'))
  for cold in (False,True):
   until=time.monotonic()+.3
   while time.monotonic()<until:
    for graph in graphs.values():sample(graph,cold)
   values={key:[]for key in ('A1','A2','B1','B2','C1','C2')};rng=random.Random(527)
   for repeat in range(10):
    order=list(values);rng.shuffle(order)
    for label in order:values[label].append(sample(graphs[label[0]],cold))
   med={k:statistics.median(v)for k,v in values.items()};ratio=lambda a,b:max(a,b)/min(a,b);controls={arm:ratio(med[arm+'1'],med[arm+'2'])for arm in 'ABC'};drift={k:ratio(statistics.median(v[:5]),statistics.median(v[5:]))for k,v in values.items()}
   paired={arm:[(values['A1'][i]+values['A2'][i])/(values[arm+'1'][i]+values[arm+'2'][i])for i in range(10)]for arm in 'BC'}
   report['timings'].append({'pattern':pattern,'cold_flush_outside_events':cold,'samples_us':values,'paired_ratios':paired,'median_paired':{k:statistics.median(v)for k,v in paired.items()},'AA_BB_CC':controls,'half_drift':drift,'qualified':max(list(controls.values())+list(drift.values()))<=1.1,'environment':'TTSuntouched, shared-system provisional timing'});save()
 memory('complete');HummingMethod.forward_layer=original_forward;report['status']='COMPLETE_LOCAL_FULL_HELPER_RESEARCH_NOT_SERVING';report['sources_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in report['source_hashes'].items());assert report['sources_unchanged'];save();print(json.dumps({'status':report['status'],'checks':len(report['checks']),'timings':len(report['timings'])}))
if __name__=='__main__':main()
