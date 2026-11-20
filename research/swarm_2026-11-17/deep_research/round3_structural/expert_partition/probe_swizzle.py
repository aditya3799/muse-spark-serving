"""Research-only same-config indexed task-order gate and whole-helper timing."""
from pathlib import Path
import argparse,ast,copy,gc,hashlib,importlib.util,json,os,random,statistics,sys,time
P=Path(__file__).resolve().parent;ROOT=P.parents[3];OLD=ROOT/'swarm_2026-11-16/moe';EXPORT=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang'
from indexed_swizzle import render_swizzle,map_task

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--tokens',nargs='+',type=int,default=[3012]);ap.add_argument('--patterns',nargs='+',default=['uniform','mixed']);ap.add_argument('--samples',type=int,default=8);ap.add_argument('--group',type=int,default=2);ap.add_argument('--experts',type=int,default=32);ap.add_argument('--reference-e32-geometry',action='store_true');a=ap.parse_args()
 if any(m<513 or m>16384 for m in a.tokens) or a.samples<8 or a.group not in (2,4,8):raise ValueError('Bounded large-prefill research scope')
 if a.experts not in (8,32) or (a.experts!=32 and not a.reference_e32_geometry):raise ValueError('Reduced-expert fixture needs explicit reference geometry label')
 assert os.environ.get('MAX_JOBS')=='1' and os.environ.get('HUMMING_DISABLE_PARALLEL_BUILD')=='1'
 a.out.mkdir(exist_ok=True,parents=True)
 for path in (OLD/'humming_vendor',OLD/'phase2',ROOT/'optimization_marlin_tiles_2026-11-15'):sys.path.insert(0,str(path))
 from kernel_runtime import bootstrap
 bootstrap(ROOT/'sglang')
 import torch
 from humming import ops
 from humming.layer import HummingMethod
 from humming.kernel.humming import HummingKernel
 from humming.jit.compiler import NVRTCCompiler
 from fixture_tuning_cache import cached_threshold_for_preparation
 from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
 props=torch.cuda.get_device_properties();assert(props.major,props.minor)==(8,6),'Local SM86 gate only'
 def module(name,path):
  spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
 scale=module('sglang.kernels.ops.moe.muse_spark_humming_scale',EXPORT/'kernels/ops/moe/muse_spark_humming_scale.py')
 final=module('swizzle_scored_finalizer',EXPORT/'kernels/ops/moe/muse_spark_post_expert_finalize_triton.py')
 src=EXPORT/'srt/layers/muse_spark/muse_spark_nvfp4_moe.py';tree=ast.parse(src.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_prepare_humming_experts','_humming_expert_outputs_default')];ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),ns)
 helper_node=next(n for n in nodes if n.name=='_humming_expert_outputs_default');helper=ast.unparse(helper_node)
 old13='cache13 = torch.empty((m * topk * max(2 * i, k),), device=x.device, dtype=x.dtype)';old2='cache2 = torch.empty((m * topk, i), device=x.device, dtype=x.dtype)'
 assert helper.count(old13)==1 and helper.count(old2)==1
 helper=helper.replace(old13,'cache13 = fixed_cache13').replace(old2,'cache2 = fixed_cache2');fixedns={'torch':torch};exec(helper,fixedns)
 (a.out/'fixed_workspace_helper.py').write_text(helper+'\n')
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 files=[Path(__file__),P/'indexed_swizzle.py',src,EXPORT/'kernels/ops/moe/muse_spark_humming_scale.py',EXPORT/'kernels/ops/moe/muse_spark_post_expert_finalize_triton.py',ROOT/'optimization_marlin_tiles_2026-11-15/kernel_runtime.py',OLD/'phase2/fixture_tuning_cache.py',*sorted((OLD/'humming_vendor/humming').rglob('*.py')),*sorted((OLD/'humming_vendor/humming/include').rglob('*.cuh'))]
 report={'status':'RUNNING','scope':'Full scored default large-M helper: alignment,W13,BF16scale,stockactivation,W2,scoredfinalizer. Same canonical weights/configs; task-order only. Shared fixed workspace substitutes exactlytwo torch.empty calls for capture memory feasibility; allocations are not GPU graph nodes. No serving/H200 claim.','source_hashes':{str(p):sha(p) for p in files},'gpu':props.name,'group':a.group,'checks':[],'kernels':[],'rows':[],'memory':[]}
 def save():
  t=a.out/'results.tmp';t.write_text(json.dumps(report,indent=2)+'\n');t.replace(a.out/'results.json')
 report['experts']=a.experts;report['reference_e32_geometry']=a.reference_e32_geometry;report['timing_environment']='External TTS remains running; concurrent-service timings provisional until isolated confirmation'
 free,total=torch.cuda.mem_get_info();weight_bytes=a.experts*27*1024**2;fixture_peak=2*weight_bytes+256*1024**2;report['fixture_memory_preflight']={'free':free,'estimated_peak_additional':fixture_peak,'reserve':512*1024**2};save()
 if fixture_peak+512*1024**2>free:report['status']='MEMORY_PREFLIGHT_BLOCKED';save();return
 stock=ROOT/'sglang/python/sglang/test/test_marlin_utils.py';gen=next(n for n in ast.parse(stock.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='make_nvfp4_weight_and_ref');gn={'torch':torch};exec(compile(ast.Module(body=[gen],type_ignores=[]),str(stock),'exec'),gn)
 report['source_hashes'][str(stock)]=sha(stock);torch.manual_seed(771);e=a.experts;h=4096;raw=torch.nn.Module()
 for stage,n in [('w13',8192),('w2',4096)]:
  ws=[];ss=[];gs=[]
  for expert in range(e):
   w,s,g,ref=gn[gen.name](n,h,torch.bfloat16);ws.append(w);ss.append(s);gs.append(g);del ref
  for suffix,arr in [('weight',ws),('weight_scale',ss),('weight_scale_2',gs)]:setattr(raw,stage+'_'+suffix,torch.nn.Parameter(torch.stack(arr),requires_grad=False))
  getattr(raw,stage+'_weight_scale_2').mul_(torch.linspace(.25,2.,e,device='cuda',dtype=torch.bfloat16));del ws,ss,gs,w,s,g,arr
 with cached_threshold_for_preparation(a.out):layer=ns['_prepare_humming_experts'](raw,torch)
 del raw;gc.collect();torch.cuda.empty_cache()
 if a.reference_e32_geometry:
  cfg_source=ROOT/'swarm_2026-11-17/deep_research/expert_partition/k_preserve_m1024_01/owned_configs.json';layer.fp32_configs=copy.deepcopy(json.loads(cfg_source.read_text())['indexed']);report['source_hashes'][str(cfg_source)]=sha(cfg_source);report['scope']+=' ReducedE fixture with frozen E32 geometry: not actual reducedE selector or nativeE32 throughput.'
 print('fixture ready',flush=True)
 original=HummingMethod.forward_layer;mode={'group':0};tables={};kept=[];audit_locks={}
 class Compiler(NVRTCCompiler):
  @classmethod
  def get_flags(cls,*args,**kw):return super().get_flags(*args,**kw)+['--ptxas-options=-v']
 def key(stage,c,group):return (stage,json.dumps(c,sort_keys=True),group)
 def prepare(stage,c,group):
  config=json.loads(layer.humming_metas[stage].to_str());config.pop('sublayer_name',None);config.update(c);sms=config.pop('num_sms',0);config.update(gemm_type='indexed',use_f16_accum=False)
  if key(stage,c,group) in tables:return
  class Kernel(HummingKernel):
   def __new__(cls,*args,**kw):obj=object.__new__(cls);obj.inited=False;obj.cubin_loaded=False;return obj
   @staticmethod
   def _get_compiler():return Compiler
   def prepare(self):
    assert self.sm_version==86 and not self.use_tma and not self.use_warp_spec
    self.code,self.research=render_swizzle(self.code,config,abs(group),audit=group<0);super().prepare()
  kernel=Kernel(**config);kept.append(kernel);tables[key(stage,c,group)]=torch.tensor([0,1<<30,kernel.kernel_id,sms],dtype=torch.int64,device='cpu')
  d=a.out/('kernel_'+str(len(kept))+'_'+stage+'_g'+str(group));d.mkdir(exist_ok=True);(d/'kernel.cu').write_text(kernel.code);cubin=Path(kernel.kernel_filename);(d/'kernel.sm86.cubin').write_bytes(cubin.read_bytes());resources=[]
  for name in ['stdout.log','stderr.log','cmdline.json']:
   q=cubin.parent/name
   if q.exists():(d/name).write_bytes(q.read_bytes());resources.extend(x for x in q.read_text(errors='replace').splitlines() if 'Used ' in x or 'spill' in x or 'stack frame' in x)
  report['kernels'].append({'stage':stage,'group':group,'configuration':config,'num_sms':sms,'native_mma_sha256':hashlib.sha256(kernel.mma_op_class.to_cpp_str().encode()).hexdigest(),'resource_lines':resources,'shared_bytes':kernel.estimated_smem_size,'cubin_sha256':sha(cubin),'research':kernel.research});save()
 def dispatch(target,**kw):
  group=mode['group']
  if group==0:return original(target,**kw)
  stage=kw['sublayer_name'];c=kw['tuning_config']
  return ops.launch_kernel(configs=tables[key(stage,c,group)],inputs=kw['inputs'],outputs=kw['outputs'],weight=getattr(target,stage+'_weight'),weight_scale=getattr(target,stage+'_weight_scale'),sorted_ids=kw['sorted_ids'],expert_ids=kw['expert_ids'],num_tokens_padded=kw['num_tokens_padded'],locks=audit_locks[stage] if group<0 else target.locks,top_k=kw['top_k'],valid_shape_m=kw['valid_shape_m'])
 HummingMethod.forward_layer=staticmethod(dispatch)
 def exact(x,y,label,**meta):
  equal=True;finite=True
  xx=x.reshape(-1);yy=y.reshape(-1)
  for lo in range(0,xx.numel(),1<<20):
   p,q=xx[lo:lo+(1<<20)],yy[lo:lo+(1<<20)];equal=equal and bool(torch.equal(p.view(torch.int16),q.view(torch.int16)));finite=finite and bool(torch.isfinite(p).all() and torch.isfinite(q).all())
  row=dict(label=label,bitwise=equal,finite=finite,**meta);report['checks'].append(row);save()
  if not(equal and finite):report['status']='FAILED_STRICT_GATE';save();raise AssertionError(row)
 def set_ids(ids,pattern,epoch):
  rng=random.Random(917+len(ids)+epoch);rows=[]
  for t in range(len(ids)):
   row=list(range((epoch%max(1,e//8))*8,(epoch%max(1,e//8)+1)*8)) if pattern=='hot8' or pattern=='mixed' and t%2==0 else rng.sample(range(256),8)
   if pattern=='remote' or epoch==2 and t>=len(ids)*2//3:row=rng.sample(range(32,256),8)
   rows.append([v if v<e else -1 for v in row])
  ids.copy_(torch.tensor(rows,device='cuda',dtype=torch.int32));return [sum(v==i for row in rows for v in row) for i in range(e)]
 norm=torch.linspace(.5,1.5,h,device='cuda',dtype=torch.bfloat16);flush=torch.empty(64*1024**2,device='cuda',dtype=torch.uint8)
 start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
 def capture(fn):
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(2):fn()
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):out=fn()
  return g,out
 def sample(graph,cold):
  if cold:flush.fill_(71)
  start.record();graph.replay();end.record();end.synchronize();return start.elapsed_time(end)*1000
 for m in a.tokens:
  R=m*8;c={s:next(copy.deepcopy(v) for lo,hi,v in layer.fp32_configs[s] if lo<R<=hi) for s in ['w13','w2']}
  assert R>4096 and all(not v['use_stream_k'] and not v['use_f16_accum'] for v in c.values())
  for stage in c:
   for group in [1,a.group,-a.group]:prepare(stage,c[stage],group)
   pair=[r for r in report['kernels'] if r['stage']==stage and r['configuration']['block_shape']==c[stage]['block_shape'] and r['configuration']['warp_shape']==c[stage]['warp_shape']]
   assert len(set(r['native_mma_sha256'] for r in pair))==1
  ids=torch.empty((m,8),device='cuda',dtype=torch.int32);x=torch.randn((m,h),device='cuda',dtype=torch.bfloat16)/20;scores=torch.rand((m,8),device='cuda');scores/=scores.sum(-1,keepdim=True)
  # Standalone raw stage gates before large shared helper workspace allocation.
  for pattern in a.patterns:
   counts=set_ids(ids,pattern,0)
   for stage,N in [('w13',8192),('w2',4096)]:
    torch.cuda.empty_cache();free,_=torch.cuda.mem_get_info();needed=2*R*N*2+(R*h*2 if stage=='w2' else 0)
    report['memory'].append({'M':m,'stage':stage,'free':free,'needed_raw_gate':needed,'margin':512*1024**2});save()
    if needed+512*1024**2>free:report.setdefault('skipped',[]).append({'M':m,'stage':stage,'scope':'raw_stage_gate','reason':'memory_preflight'});save();continue
    inp=x if stage=='w13' else torch.randn((R,h),device='cuda',dtype=torch.bfloat16)/20;out0=torch.full((R,N),37.,device='cuda',dtype=torch.bfloat16);out1=torch.full_like(out0,37.)
    ali=moe_align_block_size(ids,c[stage]['block_shape'][0],e,ignore_invalid_expert=True);mb=int(ali[2].item())//c[stage]['block_shape'][0];ex=ali[1][:mb].cpu().tolist();assert ex==sorted(ex) and all(0<=v<e for v in ex)
    groups=sum(1 for first in range(0,mb,a.group) if min(a.group,mb-first)>1 and ex[first]==ex[min(first+a.group,mb)-1]);report.setdefault('occupancy',[]).append({'M':m,'pattern':pattern,'stage':stage,'counts':counts,'m_blocks':mb,'eligible_groups':groups,'group':a.group})
    kw=dict(inputs=inp,top_k=8 if stage=='w13' else 1,sublayer_name=stage,tuning_config=c[stage],compute_config={'gemm_type':'indexed','use_f16_accum':False},sorted_ids=ali[0],expert_ids=ali[1],num_tokens_padded=ali[2],valid_shape_m=R)
    mode['group']=0;dispatch(layer,outputs=out0,**kw)
    for group in [1,a.group]:
     mode['group']=group;out1.fill_(37.);dispatch(layer,outputs=out1,**kw);exact(out0,out1,'raw_'+stage,M=m,pattern=pattern,group=group)
    del inp,out0,out1,ali;gc.collect();torch.cuda.empty_cache()
  required=3*R*h*2+4*m*h*2;free,_=torch.cuda.mem_get_info()
  if required+512*1024**2>free:report.setdefault('skipped',[]).append({'M':m,'scope':'full_helper','reason':'memory_preflight','free':free,'needed':required});save();del x,ids,scores;continue
  fixedns['fixed_cache13']=torch.empty((R*2*h,),device='cuda',dtype=torch.bfloat16);fixedns['fixed_cache2']=torch.empty((R,h),device='cuda',dtype=torch.bfloat16)
  def run():
   down,_=fixedns['_humming_expert_outputs_default'](layer,x,ids,torch,defer_downscale=True)
   return final.fused_post_expert_norm_weighted_sum(down,scores,norm,slot_expert_ids=ids,expert_output_scale=layer.w2_post_scale)
  audit_locks.clear()
  for stage,N in [('w13',8192),('w2',4096)]:audit_locks[stage]=torch.zeros(((R+c[stage]['block_shape'][0]-1)//c[stage]['block_shape'][0]+e)*(N//c[stage]['block_shape'][1])+1,device='cuda',dtype=torch.int32)
  set_ids(ids,a.patterns[0],0);graphs={};outs={}
  for group in [0,1,a.group,-a.group]:mode['group']=group;graphs[group],outs[group]=capture(run)
  assert len({v.data_ptr() for v in outs.values()})==4,'Final graph outputs must not alias'
  for pattern in a.patterns:
   row={'M':m,'pattern':pattern,'configuration':c,'timing':{}};report['rows'].append(row)
   for epoch in range(3):
    counts=set_ids(ids,pattern,epoch);x.copy_(torch.randn_like(x)/20);scores.copy_(torch.rand_like(scores));scores/=scores.sum(-1,keepdim=True)
    outs[0].fill_(37.);graphs[0].replay()
    if pattern=='remote':assert int(torch.count_nonzero(outs[0]))==0
    if epoch==2:assert int(torch.count_nonzero(outs[0][m*2//3:]))==0
    for group in [1,a.group,-a.group]:
     if group<0:
      for v in audit_locks.values():v.zero_()
     outs[group].fill_(37.);graphs[group].replay();exact(outs[0],outs[group],'full_helper_final',M=m,pattern=pattern,epoch=epoch,group=group)
     if group<0:
      for stage,N in [('w13',8192),('w2',4096)]:
       bm,bn,_=c[stage]['block_shape'];tasks=sum((count+bm-1)//bm for count in counts)*(N//bn);v=audit_locks[stage];unique=bool((v[:tasks]==1).all());tail=bool((v[tasks:]==0).all());report.setdefault('ownership_checks',[]).append({'M':m,'pattern':pattern,'epoch':epoch,'stage':stage,'tasks':tasks,'every_task_once':unique,'no_expert_cross_or_tail_write':tail});save();assert unique and tail

   row['timed_counts']=set_ids(ids,pattern,0)
   for cold in [False,True]:
    warm=[]
    for group in [0,a.group]:
     t=time.monotonic();n=0
     while time.monotonic()-t<.3:sample(graphs[group],cold);n+=1
     warm.append({'group':group,'iterations':n})
    samples={key:[] for key in ['A1','B1','A2','B2']}
    for i in range(a.samples):
     order=['A1','B1','A2','B2'] if i%2==0 else ['B2','A2','B1','A1']
     for arm in order:samples[arm].append(sample(graphs[0 if arm[0]=='A' else a.group],cold))
    med={k:statistics.median(v) for k,v in samples.items()};aa=max(med['A1'],med['A2'])/min(med['A1'],med['A2']);bb=max(med['B1'],med['B2'])/min(med['B1'],med['B2']);drift={k:max(statistics.median(v[:len(v)//2]),statistics.median(v[len(v)//2:]))/min(statistics.median(v[:len(v)//2]),statistics.median(v[len(v)//2:])) for k,v in samples.items()};paired=[samples['A'+str(j)][i]/samples['B'+str(j)][i] for i in range(a.samples) for j in [1,2]]
    row['timing']['cold' if cold else 'warm']={'samples_us':samples,'medians_us':med,'median_paired_speedup':statistics.median(paired),'paired_ratios':paired,'AA':aa,'BB':bb,'half_drift':drift,'qualified':max([aa,bb]+list(drift.values()))<=1.1,'warmup':warm};save()
   print(json.dumps({'M':m,'pattern':pattern,'timing':{k:(v['median_paired_speedup'],v['qualified']) for k,v in row['timing'].items()}}),flush=True)
  del graphs,outs,x,ids,scores;fixedns.pop('fixed_cache13');fixedns.pop('fixed_cache2');gc.collect();torch.cuda.empty_cache()
 report['status']='COMPLETED';report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();report['source_unchanged']=all(sha(Path(p))==v for p,v in report['source_hashes'].items());assert report['source_unchanged'];save()
if __name__=='__main__':main()
