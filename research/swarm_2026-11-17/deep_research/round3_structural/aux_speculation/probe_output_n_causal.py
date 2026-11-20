"""Bounded exact-operand / same-precision algorithm study, not serving integration."""
import argparse,ast,gc,hashlib,importlib.util,json,os,random,statistics,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[3];OLD=ROOT/'swarm_2026-11-16/moe';EXPORT=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--experts',type=int,nargs='+',default=[1,2]);ap.add_argument('--rows',type=int,nargs='+',default=[32,128,512,2048]);ap.add_argument('--samples',type=int,default=8);ap.add_argument('--tile-n',type=int,default=128);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 for f in (OLD/'humming_vendor',OLD/'phase2',ROOT/'optimization_marlin_tiles_2026-11-15'):sys.path.insert(0,str(f))
 from kernel_runtime import bootstrap
 bootstrap(ROOT/'sglang')
 import torch
 from fixture_tuning_cache import cached_threshold_for_preparation
 from staged_ops import dequant,scatter
 from native_materialize_tiled import TiledMaterializer
 from output_n_band_ops import scatter_n_band
 from humming import ops,dtypes
 from humming.layer import HummingMethod
 from humming.kernel.humming import HummingKernel
 from humming.jit.compiler import NVRTCCompiler
 import inspect,textwrap,copy
 from adversarial_fixture import weight_fixture,activation_fixture
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
 report={'status':'RUNNING','scope':'Local E1/E2 H4096 I4096 per-expert chain with full scored finalizer. Each selected expert processes every input row; changed slot permutation and masked routes are correctness-only. Not generic E32 dispatch or serving proof.','operand_contract':'Exact finite NVFP4 group-scaled BF16 values; global BF16 scales remain after GEMMs.','algorithm_contract':'Native BF16/FP32, fullK andBK64/WK64 retained. BM64/WM32/BN256 changes toBM16/WM16/BN128, with explicitmatchedW4control; only baseMMAprimitive hash equal, notfullgeometry. Quarterweight pluspostconversionBF16channel4 andoriginalglobalboundary retained. Strictbitwisegate required.','comparisons':[],'timings':[],'workspace_bytes':64*1024**2,'checks':[],'timing_environment':'External user-authorized VoxCPM TTS remains running; all timing is provisional even when pair controls pass.','memory_preflight':[]}
 source_paths=[Path(__file__),P/'native_materialize_tiled.py',P/'native_materialize_tiled.cuh',P/'output_n_band_ops.py',P/'adversarial_fixture.py',P/'staged_ops.py',src,stock,EXPORT/'python/sglang/kernels/ops/moe/muse_spark_humming_split.py',EXPORT/'python/sglang/kernels/ops/moe/muse_spark_humming_scale.py',EXPORT/'python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py',ROOT/'sglang/python/sglang/kernels/jit/csrc/elementwise/activation.cuh',ROOT/'swarm_2026-11-17/deep_research/expert_partition/k_preserve_m1024_01/owned_configs.json',OLD/'humming_vendor/humming/kernel/humming.py']
 report['source_hashes']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths};own_budget=int(1.15*1024**3);report['own_peak_budget_bytes']=own_budget
 def save():(args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
 def compare(a,b,**meta):
  a=a.contiguous().view(-1);b=b.contiguous().view(-1);finite=True;bits=True;max_abs=0.;sum_sq=0.;max_rel=0.;mismatch_samples=[]
  for lo in range(0,a.numel(),1<<20):
   aa=a[lo:lo+(1<<20)];bb=b[lo:lo+(1<<20)];finite=finite and bool(torch.isfinite(aa).all() and torch.isfinite(bb).all());bits=bits and torch.equal(aa.view(torch.int16),bb.view(torch.int16))
   if len(mismatch_samples)<16:
    locations=(aa.view(torch.int16)!=bb.view(torch.int16)).nonzero().flatten()[:16-len(mismatch_samples)].tolist()
    mismatch_samples.extend({'index':lo+i,'candidate':float(aa[i]),'reference':float(bb[i]),'candidate_bits':int(aa.view(torch.int16)[i]),'reference_bits':int(bb.view(torch.int16)[i])} for i in locations)
   err=(aa.float()-bb.float()).abs();max_abs=max(max_abs,float(err.max()));sum_sq+=float(err.square().sum());max_rel=max(max_rel,float((err/bb.float().abs().clamp_min(1e-6)).max()))
  report['comparisons'].append(dict(meta,elements=a.numel(),finite=finite,bitwise=bits,mismatch_samples=mismatch_samples,max_abs=max_abs,rms_abs=(sum_sq/max(a.numel(),1))**.5,max_relative_to_1e_minus6_floor=max_rel));save()
  if not (finite and bits):report['status']='FAILED_STRICT_NATIVE_GATE';save();raise AssertionError(meta)

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
 assert args.experts==[1] and args.rows==[512] and args.tile_n==128, 'SingleE1M512 N128/BM16 matched-control causalgate'
 for e in args.experts:
  torch.manual_seed(799+e);raw=torch.nn.Module()
  for stage,n in (('w13',2*h),('w2',h)):
   ws=[];ss=[];gg=[]
   for expert in range(e):
    w,s,g=weight_fixture(n,h,stage,expert,0,device='cuda');ws.append(w);ss.append(s);gg.append(g)
   for suffix,values in (('weight',ws),('weight_scale',ss),('weight_scale_2',gg)):setattr(raw,stage+'_'+suffix,torch.nn.Parameter(torch.stack(values),requires_grad=False))
   del ws,ss,gg,w,s,g,values
  with cached_threshold_for_preparation(args.out):baseline=ns['_prepare_humming_experts'](raw,torch)
  frozen=ROOT/'swarm_2026-11-17/deep_research/expert_partition/k_preserve_m1024_01/owned_configs.json'
  baseline.fp32_configs=copy.deepcopy(json.loads(frozen.read_text())['indexed'])
  report['frozen_geometry_source']=str(frozen);report['frozen_geometry_sha256']=hashlib.sha256(frozen.read_bytes()).hexdigest()
  report['scope']='OnecausalE1M512 N128/BM16/WM16/BN128 B16candidate, fullK4096/BK64/WK64 retained. Actualscoredadaptive andfullNW4matchedgeometry comparators.1MiBweighttile nominallyfits1.5MiBL2 butinputs/outputs compete;32logicalMtiles vsoriginal8; schedulergrid/occupancyunmeasured. Allmaterialization/GEMM/scatter/fullnativeactivation/finalizer measured. NoE32/H200/servingclaim.'
  mode={'packed':False,'events':None};tables={};kept=[];channel4={};materializers={};native_forward=HummingMethod.forward_layer
  checked=textwrap.dedent(inspect.getsource(HummingKernel.check_dtype));cap='assert self.b_dtype.num_bits <= 8';assert checked.count(cap)==1
  checked=checked.replace(cap,'assert self.b_dtype.num_bits <= 8 or (self.a_dtype == self.b_dtype == self.c_dtype == self.bs_dtype == dtypes.bfloat16 and self.is_channel_weight_scale and not self.has_zero_point and not self.has_bias and not self.use_f16_accum)')
  cns={'dtypes':dtypes};exec(compile(checked,'research_dtype','exec'),cns)
  class Compiler(NVRTCCompiler):
   @classmethod
   def get_flags(cls,*a,**kw):return super().get_flags(*a,**kw)+['--ptxas-options=-v']
  class Kernel(HummingKernel):
   def __new__(cls,*a,**kw):obj=object.__new__(cls);obj.inited=False;obj.cubin_loaded=False;return obj
   check_dtype=cns['check_dtype']
   @staticmethod
   def _get_compiler():return Compiler
   def prepare(self):
    assert self.sm_version==86 and not self.use_tma and not self.use_warp_spec and self.estimated_smem_size<=torch.cuda.get_device_properties().shared_memory_per_block_optin
    super().prepare()
  def measured(label,fn,*args,**kwargs):
   pair=mode['events'].get(label) if mode['events'] is not None else None
   if pair is not None:pair[0].record()
   result=fn(*args,**kwargs)
   if pair is not None:pair[1].record()
   return result
  def dispatch(target,**kw):
   if not mode['packed']:return native_forward(target,**kw)
   stage=kw['sublayer_name'];n=2*h if stage=='w13' else h
   for offset in range(0,n,args.tile_n):
    measured(f'{stage}:materialize:{offset}',materializers[stage],getattr(target,stage+'_weight')[0],getattr(target,stage+'_weight_scale')[0],weight_workspace,row_offset=offset)
    measured(f'{stage}:gemm:{offset}',ops.launch_kernel,configs=tables[stage],inputs=kw['inputs'],outputs=tile_output,weight=weight_workspace.view(torch.int32).reshape(1,h//16,args.tile_n*8),weight_scale=channel4[stage],sorted_ids=kw['sorted_ids'],expert_ids=kw['expert_ids'],num_tokens_padded=kw['num_tokens_padded'],locks=target.locks,top_k=kw['top_k'],valid_shape_m=kw['valid_shape_m'])
    measured(f'{stage}:scatter:{offset}',scatter_n_band,tile_output,ids,kw['outputs'],offset)
   return kw['outputs']
  HummingMethod.forward_layer=staticmethod(dispatch)
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
  del weight_workspace;weight_workspace=torch.empty((args.tile_n,h),device='cuda',dtype=torch.bfloat16);report['workspace_bytes']=weight_workspace.numel()*2
  for stage,n in (('w13',2*h),('w2',h)):
   print('compiling_materializer_'+stage,flush=True);mat=TiledMaterializer(n,args.tile_n,h,source_transpose=False,target_transpose=False,fixed_grid_loop=False);materializers[stage]=mat
   dest=args.out/('materializer_'+stage);dest.mkdir(exist_ok=True);(dest/'kernel.cu').write_text(mat.code);cubin=Path(mat.kernel_filename);(dest/'kernel.sm86.cubin').write_bytes(cubin.read_bytes())
   for logfile in ('stdout.log','stderr.log','cmdline.json'):
    path=cubin.parent/logfile
    if path.exists():(dest/logfile).write_bytes(path.read_bytes())
   report.setdefault('materializers',[]).append({'stage':stage,'sourceN':n,'targetN':args.tile_n,'K':h,'blocks':mat.blocks,'threads':mat.threads,'generated_source_sha256':mat.source_sha256,'cubin_sha256':hashlib.sha256(cubin.read_bytes()).hexdigest()});save()
  report['launch_accounting']={'w13_tiles':8192//args.tile_n,'w2_tiles':4096//args.tile_n,'materialize_gemm_scatter_launches':3*(8192+4096)//args.tile_n,'additional':'Originalroutealignment,globalBF16scale,nativeactivation,finalizer; rawobservationcloneinbothnumericgraphs only.'}

  for rows in args.rows:
   gc.collect();torch.cuda.empty_cache();free,total=torch.cuda.mem_get_info();T=rows*h*2;S=rows*topk*h*2
   estimated_additional=9*S+8*T+2*e*T+128*1024**2;reserve=512*1024**2
   report['memory_preflight'].append({'experts':e,'rows':rows,'free_bytes':free,'total_bytes':total,'estimated_additional_peak':estimated_additional,'required_reserve':reserve,'fits':free>=estimated_additional+reserve});save()
   if free<estimated_additional+reserve:
    report.setdefault('skipped',[]).append({'experts':e,'rows':rows,'reason':'External workload plus512MiB reserve leaves insufficient estimated memory'});save();continue
   matched_layer=copy.copy(baseline);matched_layer.fp32_configs={}
   for stage in ('w13','w2'):
    original_c=next(copy.deepcopy(v) for lo,hi,v in baseline.fp32_configs[stage] if lo<rows*8<=hi)
    c=copy.deepcopy(original_c);c['block_shape']=[16,128,original_c['block_shape'][2]];c['warp_shape']=[16,original_c['warp_shape'][1],original_c['warp_shape'][2]]
    assert c['block_shape'][2]==c['warp_shape'][2]==64 and c['warp_shape'][1]==64
    matched_layer.fp32_configs[stage]=[(0,1<<30,c)]
   report['geometry_contract']={'original_selected':{stage:next(v for lo,hi,v in baseline.fp32_configs[stage] if lo<rows*8<=hi) for stage in ('w13','w2')},'matched_selected':matched_layer.fp32_configs,'warning':'BM64/WM32/BN256 changes toBM16/WM16/BN128; samebaseMMAprimitive andKtopology, notsamefullgeometry. MatchedW4control required.1MiBweighttile fitsnominalL2 only, noresidencyguarantee.32logical Mtiles vs8forfullactiveE1M512; actualschedulergrid/residentoccupancy unmeasured.'};save()
   for stage,n in (('w13',2*h),('w2',h)):
    c=copy.deepcopy(matched_layer.fp32_configs[stage][0][2]);sms=c.pop('num_sms',0)
    meta=json.loads(baseline.humming_metas[stage].to_str());meta.pop('sublayer_name',None);meta.update(c,gemm_type='indexed',use_f16_accum=False)
    native=Kernel(**meta);candidate_config=dict(meta,shape_n=args.tile_n,b_dtype='bfloat16',bs_dtype='bfloat16',weight_scale_type='channel',weight_scale_group_size=0,num_stages=2)
    print('compiling_candidate_M%d_%s'%(rows,stage),flush=True);candidate=Kernel(**candidate_config);assert native.mma_op_class.to_cpp_str()==candidate.mma_op_class.to_cpp_str();kept.extend((native,candidate))
    dest=args.out/('candidate_M%d_%s'%(rows,stage));dest.mkdir(exist_ok=True);(dest/'kernel.cu').write_text(candidate.code);cubin=Path(candidate.kernel_filename);(dest/'kernel.sm86.cubin').write_bytes(cubin.read_bytes())
    for logfile in ('stdout.log','stderr.log','cmdline.json'):
     path=cubin.parent/logfile
     if path.exists():(dest/logfile).write_bytes(path.read_bytes())
    tables[stage]=torch.tensor([0,1<<30,candidate.kernel_id,sms],dtype=torch.int64,device='cpu');channel4[stage]=torch.full((1,1,args.tile_n),4.,device='cuda',dtype=torch.bfloat16)
    report.setdefault('kernels',[]).append({'stage':stage,'original_config':meta,'candidate_config':candidate_config,'native_mma_hash':hashlib.sha256(native.mma_op_class.to_cpp_str().encode()).hexdigest(),'candidate_shared_bytes':candidate.estimated_smem_size,'candidate_cubin_sha256':hashlib.sha256(Path(candidate.kernel_filename).read_bytes()).hexdigest()});save()
   tile_output=torch.empty((rows*8,args.tile_n),device='cuda',dtype=torch.bfloat16)
   x=torch.empty((rows,h),device='cuda',dtype=torch.bfloat16);ids=torch.empty((rows,topk),device='cuda',dtype=torch.int32);scores=torch.empty((rows,topk),device='cuda')
   gateup=torch.empty((rows,2*h),device='cuda',dtype=torch.bfloat16);activation=torch.empty((rows,h),device='cuda',dtype=torch.bfloat16);tempdown=torch.empty_like(activation)
   scale_ids=[torch.full((rows,),expert,device='cuda',dtype=torch.int32) for expert in range(e)]
   def inputs(generation):
    torch.manual_seed(291+generation);x.copy_(activation_fixture(rows,h,generation,device='cuda'));scores.copy_(torch.rand_like(scores));scores.div_(scores.sum(-1,keepdim=True));ids.fill_(-1)
    for expert in range(e):ids[:,(expert+generation)%topk]=expert
    if generation==2:ids[::3,:]=-1
    if generation==3:ids.fill_(-1)
   def finish(down):return finalmod.fused_post_expert_norm_weighted_sum(down,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=ids,expert_output_scale=baseline.w2_post_scale)
   observed=[];observed_raw=[]
   def observer(x,out,*a,**kw):
    observed_raw[:]=[x.clone()];result=native_act(x,out,*a,**kw);observed[:]=[out];return result
   def original():
    actmod.silu_and_mul=observer
    try:down,local=ns['_humming_expert_outputs_default'](matched_layer if mode['packed'] else baseline,x,ids,torch,defer_downscale=True)
    finally:actmod.silu_and_mul=native_act
    return down,observed[-1],finish(down),observed_raw[-1]
   def staged(debug=False):
    mode['packed']=True
    try:
     down,act,final,raw13=original();return down,act.view(rows,topk,h),final,raw13
    finally:mode['packed']=False
   print('capturing_M%d'%rows,flush=True);inputs(0);ga,oa=capture(original);gb,ob=capture(staged)
   assert len({oa[2].data_ptr(),ob[2].data_ptr()})==2
   for generation in range(4):
    if generation>0:
     # Change codes, group scales and globals at fixed captured pointers.
     for stage,n in (('w13',2*h),('w2',h)):
      for expert in range(e):
       w,s,g=weight_fixture(n,h,stage,expert,generation,device='cuda')
       getattr(raw,stage+'_weight')[expert].copy_(w)
       getattr(raw,stage+'_weight_scale')[expert].copy_(s)
       getattr(raw,stage+'_weight_scale_2')[expert].copy_(g)
       del w,s,g
     fresh=ns['_prepare_humming_experts'](raw,torch)
     assert all(getattr(fresh,stage+'_weight').shape==getattr(baseline,stage+'_weight').shape for stage in ('w13','w2'))
     for stage in ('w13','w2'):
      for suffix in ('weight','weight_scale','post_scale'):
       getattr(baseline,stage+'_'+suffix).copy_(getattr(fresh,stage+'_'+suffix))
     del fresh
    print('checking_M%d_generation%d'%(rows,generation),flush=True);inputs(generation)
    for output in (oa,ob):output[0].fill_(37.);output[2].fill_(37.)
    ga.replay();gb.replay();torch.cuda.synchronize();active=ids>=0
    invalid=~active.any(dim=-1)
    zero_ok=all(bool((output[2][invalid]==0).all()) for output in (oa,ob))
    report['checks'].append({'generation':generation,'kind':'invalid_rows_zero','invalid_rows':int(invalid.sum()),'pass':zero_ok});save();assert zero_ok
    if bool(active.any()):
     compare(ob[3][active.flatten()],oa[3][active.flatten()],e=e,rows=rows,generation=generation,stage='scaled_W13')
     compare(ob[1][active],oa[1][active.flatten()],e=e,rows=rows,generation=generation,stage='activation')
     compare(ob[0][active],oa[0][active],e=e,rows=rows,generation=generation,stage='down')
    compare(ob[2],oa[2],e=e,rows=rows,generation=generation,stage='final')
   inputs(0);gb.replay();torch.cuda.synchronize();bb=ob[2].clone();ob[2].fill_(37.);gb.replay();torch.cuda.synchronize();compare(ob[2],bb,e=e,rows=rows,stage='candidate_repeat_control');del bb
   inputs(0);ga.replay();torch.cuda.synchronize();aa=oa[2].clone();ga.replay();torch.cuda.synchronize()
   compare(oa[2],aa,e=e,rows=rows,stage='baseline_repeat_control');assert torch.equal(oa[2].view(torch.int16),aa.view(torch.int16));del aa
   # Release numeric observation graphs before allocating clean comparators.
   observed.clear();observed_raw.clear();del ga,gb,oa,ob;gc.collect();torch.cuda.empty_cache()
   if rows==512:
    free,total=torch.cuda.mem_get_info();needed=16*S+128*1024**2
    live=torch.cuda.memory_allocated();report['clean_timing_memory_preflight']={'free':free,'live_allocated':live,'reserved':torch.cuda.memory_reserved(),'prior_peak':torch.cuda.max_memory_allocated(),'additional_budget':needed,'own_peak_budget':own_budget,'reserve':512*1024**2};save()
    if free<needed+512*1024**2 or live+needed>own_budget:report['status']='HELD_CLEAN_TIMING_MEMORY';save();return
    print('capturing_clean_causal_M512',flush=True);clean_graphs={};clean_outputs={}
    labels=[f'{stage}:{kind}:{off}' for stage,n in (('w13',8192),('w2',4096)) for off in range(0,n,args.tile_n) for kind in ('materialize','gemm','scatter')]+['finalizer']
    profile_events={label:(torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)) for label in labels}
    for first,last in profile_events.values():first.record();last.record()
    torch.cuda.synchronize()
    def clean(arm):
     mode['packed']=arm in ('B','P');mode['events']=profile_events if arm=='P' else None
     try:
      if arm=='C':down,local=ns['_humming_expert_outputs'](baseline,x,ids,torch,defer_downscale=True,global_experts=256)
      else:down,local=ns['_humming_expert_outputs_default'](matched_layer if arm in ('B','D','P') else baseline,x,ids,torch,defer_downscale=True)
      return measured('finalizer',finish,down)
     finally:mode['packed']=False;mode['events']=None
    inputs(0)
    for arm in ('A','B','C','D','P'):
     free,_=torch.cuda.mem_get_info();live=torch.cuda.memory_allocated();additional=3*S+32*1024**2
     if free<additional+512*1024**2 or live+additional>own_budget:report['status']='HELD_COMPARATOR_MEMORY';save();return
     clean_graphs[arm],clean_outputs[arm]=capture(lambda arm=arm:clean(arm))
     record={'arm':arm,'live_allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated(),'free':torch.cuda.mem_get_info()[0]};report.setdefault('comparator_memory',[]).append(record);save()
     if record['peak_allocated']>own_budget or record['free']<512*1024**2:report['status']='HELD_OBSERVED_MEMORY_BOUND';save();return
    assert len({value.data_ptr() for value in clean_outputs.values()})==5
    for generation in (0,1,3):
     inputs(generation)
     for value in clean_outputs.values():value.fill_(37.)
     for graph in clean_graphs.values():graph.replay()
     torch.cuda.synchronize()
     for arm,label in (('A','original_default'),('C','scored_adaptive'),('D','W4_matched_BM16_BN128'),('P','profiled_tiled')):compare(clean_outputs['B'],clean_outputs[arm],rows=rows,generation=generation,stage='clean_tiled_vs_'+label)
    inputs(0)
    for cold in (False,True):
     until=time.monotonic()+.3
     while time.monotonic()<until:
      for graph in clean_graphs.values():sample(graph,cold)
     values={key:[] for key in ('A1','A2','B1','B2','C1','C2','D1','D2','P')};phases=[];rng=random.Random(491+int(cold))
     for repeat in range(8):
      order=list(values);rng.shuffle(order)
      for arm in order:
       elapsed=sample(clean_graphs[arm[0]],cold);values[arm].append(elapsed)
       if arm=='P':
        record={}
        for label,(first,last) in profile_events.items():
         key=':'.join(label.split(':')[:2]);record[key]=record.get(key,0.)+first.elapsed_time(last)*1000
        record['unattributed_including_event_overhead']=elapsed-sum(record.values());phases.append(record)
     med={key:statistics.median(v) for key,v in values.items()};ratio=lambda a,b:max(a,b)/min(a,b);controls={arm+arm:ratio(med[arm+'1'],med[arm+'2']) for arm in 'ABCD'};drift={key:ratio(statistics.median(v[:4]),statistics.median(v[4:])) for key,v in values.items()};combined={arm:statistics.median(values[arm+'1']+values[arm+'2']) for arm in 'ABCD'}
     report['timings'].append({'rows':rows,'tileN':args.tile_n,'cold_sweep':cold,'arms':{'A':'originaldefault_frozenE32','B':'completeN128_BM16_B16tiled','C':'actualscoredadaptive_E1_frozenE32','D':'fullNW4_matchedBM16_BN128_BK64_WK64','P':'instrumentedB'},'samples_us':values,'median_us':med,'controls':controls,'half_drift':drift,'qualified':max(list(controls.values())+list(drift.values()))<=1.1,'speedup_vs_default':combined['A']/combined['B'],'speedup_vs_adaptive':combined['C']/combined['B'],'speedup_vs_matched_W4':combined['D']/combined['B'],'profile_over_clean':med['P']/combined['B'],'profile_overhead_qualified':med['P']/combined['B']<=1.1,'phase_samples_us':phases,'phase_median_us':{key:statistics.median(v[key] for v in phases) for key in phases[0]},'environment':'ExternalTTSunchanged; localreducedfixture only. NoE32/H200/servingclaim.'});save()
    del clean_graphs,clean_outputs,profile_events;gc.collect();torch.cuda.empty_cache()
   print(json.dumps({'e':e,'rows':rows,'timings':report['timings'][-2:]}),flush=True)
   observed.clear();observed_raw.clear();del x,ids,scores,gateup,activation,tempdown,scale_ids,tile_output,matched_layer;gc.collect();torch.cuda.empty_cache()
  HummingMethod.forward_layer=native_forward
  del raw,baseline;torch.cuda.empty_cache()
 report.update(status='OUTPUT_N_CAUSAL_COMPLETE_NOT_SERVING_QUALIFIED',strict_bitwise_gate=all(row['bitwise'] and row['finite'] for row in report['comparisons']),peak_allocated=torch.cuda.max_memory_allocated(),device=torch.cuda.get_device_name(),source_unchanged=all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==value for path,value in report['source_hashes'].items()));
 if not report['strict_bitwise_gate']:report['status']='HELD_STRICT_BITWISE_FAILURE'
 save()
if __name__=='__main__':main()
