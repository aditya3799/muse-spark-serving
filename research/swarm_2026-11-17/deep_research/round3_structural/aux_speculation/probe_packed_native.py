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
 from packed_staged_ops import packed_dequant
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
 report={'status':'RUNNING','scope':'Local E1/E2 H4096 I4096 per-expert chain with full scored finalizer. Each selected expert processes every input row; changed slot permutation and masked routes are correctness-only. Not generic E32 dispatch or serving proof.','operand_contract':'Exact finite NVFP4 group-scaled BF16 values; global BF16 scales remain after GEMMs.','algorithm_contract':'Native Humming BF16 inputs/outputs and FP32 accumulation. Matching original MMA source hash and block/warpK geometry; only pipeline3to2 and storage/dequant materialization differ. Quarterweight thenBF16channel4 preserves pinned exponent boundary. Requires strict bitwise native gate; not serving proof.','comparisons':[],'timings':[],'workspace_bytes':64*1024**2,'checks':[],'timing_environment':'External user-authorized VoxCPM TTS remains running; all timing is provisional even when pair controls pass.','memory_preflight':[]}
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
 assert args.experts==[1] and args.rows==[64], 'First packed native gate: E1/R512 with frozen scoredE32geometry'
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
  report['scope']='Research B16 capability extension. Scored default helper E1 with frozen scoredE32geometry R512, same native MMA/K geometry and same math; candidate pipeline stages2 fitsSM86. Direct packed weight quartervalues and BF16channel4. Not pinned supportedB16 API, not generic staging orH200 performance.'
  mode={'packed':False};tables={};kept=[];channel4={};native_forward=HummingMethod.forward_layer
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
  def dispatch(target,**kw):
   if not mode['packed']:return native_forward(target,**kw)
   stage=kw['sublayer_name'];n=2*h if stage=='w13' else h;out=weight_workspace[:n]
   packed_dequant(getattr(target,stage+'_weight')[0],getattr(target,stage+'_weight_scale')[0],out,n=n,k=h,source_transpose=False,target_transpose=False)
   return ops.launch_kernel(configs=tables[stage],inputs=kw['inputs'],outputs=kw['outputs'],weight=out.view(torch.int32).reshape(1,h//16,n*8),weight_scale=channel4[stage],sorted_ids=kw['sorted_ids'],expert_ids=kw['expert_ids'],num_tokens_padded=kw['num_tokens_padded'],locks=target.locks,top_k=kw['top_k'],valid_shape_m=kw['valid_shape_m'])
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
  for rows in args.rows:
   gc.collect();torch.cuda.empty_cache();free,total=torch.cuda.mem_get_info();T=rows*h*2;S=rows*topk*h*2
   estimated_additional=6*S+8*T+2*e*T+96*1024**2;reserve=512*1024**2
   report['memory_preflight'].append({'experts':e,'rows':rows,'free_bytes':free,'total_bytes':total,'estimated_additional_peak':estimated_additional,'required_reserve':reserve,'fits':free>=estimated_additional+reserve});save()
   if free<estimated_additional+reserve:
    report.setdefault('skipped',[]).append({'experts':e,'rows':rows,'reason':'External workload plus512MiB reserve leaves insufficient estimated memory'});save();continue
   for stage,n in (('w13',2*h),('w2',h)):
    c=next(copy.deepcopy(v) for lo,hi,v in baseline.fp32_configs[stage] if lo<rows*8<=hi);sms=c.pop('num_sms',0)
    meta=json.loads(baseline.humming_metas[stage].to_str());meta.pop('sublayer_name',None);meta.update(c,gemm_type='indexed',use_f16_accum=False)
    native=Kernel(**meta);candidate_config=dict(meta,b_dtype='bfloat16',bs_dtype='bfloat16',weight_scale_type='channel',weight_scale_group_size=0,num_stages=2)
    candidate=Kernel(**candidate_config);assert native.mma_op_class.to_cpp_str()==candidate.mma_op_class.to_cpp_str();kept.extend((native,candidate))
    tables[stage]=torch.tensor([0,1<<30,candidate.kernel_id,sms],dtype=torch.int64,device='cpu');channel4[stage]=torch.full((1,1,n),4.,device='cuda',dtype=torch.bfloat16)
    report.setdefault('kernels',[]).append({'stage':stage,'original_config':meta,'candidate_config':candidate_config,'native_mma_hash':hashlib.sha256(native.mma_op_class.to_cpp_str().encode()).hexdigest(),'candidate_shared_bytes':candidate.estimated_smem_size,'candidate_cubin_sha256':hashlib.sha256(Path(candidate.kernel_filename).read_bytes()).hexdigest()});save()
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
    try:down,local=ns['_humming_expert_outputs_default'](baseline,x,ids,torch,defer_downscale=True)
    finally:actmod.silu_and_mul=native_act
    return down,observed[-1],finish(down),observed_raw[-1]
   def staged(debug=False):
    mode['packed']=True
    try:
     down,act,final,raw13=original();return down,act.view(rows,topk,h),final,raw13
    finally:mode['packed']=False
   inputs(0);ga,oa=capture(original);gb,ob=capture(staged);gd,od=capture(lambda:staged(True))
   assert len({oa[2].data_ptr(),ob[2].data_ptr(),od[2].data_ptr()})==3
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
    inputs(generation)
    for output in (oa,ob,od):output[0].fill_(37.);output[2].fill_(37.)
    ga.replay();gb.replay();gd.replay();torch.cuda.synchronize();active=ids>=0
    invalid=~active.any(dim=-1)
    zero_ok=all(bool((output[2][invalid]==0).all()) for output in (oa,ob,od))
    report['checks'].append({'generation':generation,'kind':'invalid_rows_zero','invalid_rows':int(invalid.sum()),'pass':zero_ok});save();assert zero_ok
    if bool(active.any()):
     compare(ob[3][active.flatten()],oa[3][active.flatten()],e=e,rows=rows,generation=generation,stage='scaled_W13')
     compare(od[1][active],oa[1][active.flatten()],e=e,rows=rows,generation=generation,stage='activation')
     compare(ob[0][active],oa[0][active],e=e,rows=rows,generation=generation,stage='down')
    compare(ob[2],oa[2],e=e,rows=rows,generation=generation,stage='final')
    compare(ob[2],od[2],e=e,rows=rows,generation=generation,stage='debug_fast_control');assert torch.equal(ob[2].view(torch.int16),od[2].view(torch.int16))
   inputs(0);ga.replay();torch.cuda.synchronize();aa=oa[2].clone();ga.replay();torch.cuda.synchronize()
   compare(oa[2],aa,e=e,rows=rows,stage='baseline_repeat_control');assert torch.equal(oa[2].view(torch.int16),aa.view(torch.int16));del aa
   for cold in (): # Numerical challenge only; no repeated performance sweep.
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
  HummingMethod.forward_layer=native_forward
  del raw,baseline;torch.cuda.empty_cache()
 report.update(status='PACKED_NATIVE_EXPERIMENT_COMPLETE_NOT_SERVING_QUALIFIED',strict_bitwise_gate=all(row['bitwise'] and row['finite'] for row in report['comparisons']),peak_allocated=torch.cuda.max_memory_allocated(),device=torch.cuda.get_device_name());
 if not report['strict_bitwise_gate']:report['status']='HELD_STRICT_BITWISE_FAILURE'
 save()
if __name__=='__main__':main()
