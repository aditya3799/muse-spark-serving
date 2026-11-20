"""Prerequisites for an interleaved W13 scale/SiLU GEMM epilogue, not the fusion itself."""
import ast,hashlib,json,pathlib,sys,os,argparse,random,statistics,importlib.util,time
ROOT=pathlib.Path(__file__).resolve().parents[3]; OWN=pathlib.Path(__file__).resolve().parent
OLD=ROOT/'swarm_2026-11-16/moe'; EXPORT=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources'

def cpu_contract():
 cases=[]
 for hidden in (64,128,512,4096):
  perm=[v for i in range(hidden) for v in (i,i+hidden)]
  assert sorted(perm)==list(range(2*hidden))
  for bn in (64,128,256):
   if 2*hidden%bn:continue
   for bm in (8,16,24,32,40,48,64,128,176):
    for splits in (1,2):
     if splits==2 and bm%32:continue
     seen=set()
     for split in range(splits):
      for offset in range(bm*bn//8//splits):
       sr,sc=divmod(offset,8)
       gr=sr%(bm//splits)+(bm//2*split if splits==2 else 0)
       gc=sr//(bm//splits)*8+sc
       for nb in range(2*hidden//bn):
        index=nb*(bn//8)+gr*(2*hidden//8)+gc
        for pair in range(4):
         output=index*4+pair
         assert output==gr*hidden+nb*(bn//2)+gc*4+pair
         original_col=(nb*bn+gc*8+pair*2)//2
         assert perm[nb*bn+gc*8+pair*2]==original_col
         assert perm[nb*bn+gc*8+pair*2+1]==original_col+hidden
         assert output not in seen
         seen.add(output)
     assert len(seen)==bm*hidden
     cases.append({'I':hidden,'BM':bm,'BN':bn,'write_splits':splits,'outputs':len(seen)})
 return {'status':'CPU_LAYOUT_CONTRACT_PASS','cases':cases,'checks':['gate/up permutation bijective','all output addresses exactly once','int4 input to uint2 output indexing halves physical stride','both write-split cases preserve row placement'],'limits':['Does not verify GPU shared-memory values or new epilogue code','Indexed invalid-row predicates must remain unchanged','Requires custom launcher output shape N/2; stock launcher rejects it','Only no-StreamK, no-TMA-output, aligned BF16 indexed W13 is proposed']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');p.add_argument('--fusion',action='store_true');p.add_argument('--out',required=True);args=p.parse_args();out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
 report=cpu_contract();(out/'layout_contract.json').write_text(json.dumps(report,indent=2)+'\n')
 if not args.gpu:print(json.dumps({'status':report['status'],'cases':len(report['cases'])}));return
 assert os.environ.get('MAX_JOBS')=='1' and os.environ.get('HUMMING_DISABLE_PARALLEL_BUILD')=='1'
 for path in (OLD/'humming_vendor',OLD/'phase2',ROOT/'optimization_marlin_tiles_2026-11-15'):sys.path.insert(0,str(path))
 from kernel_runtime import bootstrap
 bootstrap(ROOT/'sglang')
 import torch
 from humming.layer import HummingMethod
 from fixture_tuning_cache import cached_threshold_for_preparation
 from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
 from sglang.kernels.ops.activation.activation import silu_and_mul
 scale_path=EXPORT/'python/sglang/kernels/ops/moe/muse_spark_humming_scale.py'
 spec=importlib.util.spec_from_file_location('research_scale',scale_path);scale_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(scale_module)
 scale_active_rows=scale_module.scale_active_rows
 from sglang.kernels.ops.moe.muse_spark_post_expert_finalize_triton import fused_post_expert_norm_weighted_sum
 norm=torch.linspace(.5,1.5,4096,device='cuda',dtype=torch.bfloat16)
 flush=torch.zeros((64*1024**2//4,),device='cuda',dtype=torch.int32)
 def capture(fn):
  for _ in range(2):fn()
  torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):fn()
  return graph
 def sample(graph,cold):
  if cold:flush.add_(1)
  start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();graph.replay();end.record();end.synchronize();return start.elapsed_time(end)*1000
 src=EXPORT/'python/sglang/srt/layers/muse_spark/muse_spark_nvfp4_moe.py'
 tree=ast.parse(src.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_prepare_humming_experts');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(src),'exec'),ns)
 stock=ROOT/'sglang/python/sglang/test/test_marlin_utils.py';tree=ast.parse(stock.read_text());gen=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_nvfp4_weight_and_ref');gs={'torch':torch};exec(compile(ast.Module(body=[gen],type_ignores=[]),str(stock),'exec'),gs)
 torch.manual_seed(771);h=4096;e=4;raw=torch.nn.Module()
 for prefix,n in (('w13',2*h),('w2',h)):
  weights=[];scales=[];globals_=[]
  for expert in range(e):
   w,s,g,ref=gs[gen.name](n,h,torch.bfloat16);weights.append(w);scales.append(s);globals_.append(g);del ref
  for suffix,values in (('weight',weights),('weight_scale',scales),('weight_scale_2',globals_)):
   setattr(raw,prefix+'_'+suffix,torch.nn.Parameter(torch.stack(values),requires_grad=False))
  del weights,scales,globals_,w,s,g,values
 for stage in ('w13','w2'):
  getattr(raw,stage+'_weight_scale_2').mul_(torch.tensor([.25,.5,1.,2.],device='cuda',dtype=torch.bfloat16))
 perm=torch.stack((torch.arange(h,device='cuda'),torch.arange(h,2*h,device='cuda')),dim=1).flatten();inverse=torch.argsort(perm)
 with cached_threshold_for_preparation(out):
  baseline=ns[fn.name](raw,torch)
  raw.w13_weight=torch.nn.Parameter(raw.w13_weight[:,perm].contiguous(),requires_grad=False)
  # Index bytes because native float8 advanced indexing is not universal.
  raw.w13_weight_scale=torch.nn.Parameter(raw.w13_weight_scale.view(torch.uint8)[:,perm].contiguous().view(torch.float8_e4m3fn),requires_grad=False)
  candidate=ns[fn.name](raw,torch)
 del raw
 assert baseline.fp32_configs==candidate.fp32_configs
 report={'source_dependencies':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (OWN/'w13_fused_epilogue.py',scale_path)},'status':'RUNNING','scope':'Local fused W13 epilogue, subsequent W2/finalizer numerical validation and controlled component timings; not full-model or serving' ,'geometry':{'H':h,'I':h,'local_experts':e,'top_k':8},'device':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'checks':[]}
 def save():(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
 def check(a,b,**meta):
  bitwise=torch.equal(a.contiguous().view(torch.int16),b.contiguous().view(torch.int16));delta=a.float()-b.float()
  row={**meta,'bitwise':bitwise,'max_abs':float(delta.abs().max()) if a.numel() else 0};report['checks'].append(row);save()
  if not bitwise:report['status']='FAILED_PERMUTATION_GATE';save();raise AssertionError(row)
 for m in (8,32,128,768,1024):
  for pattern in ('uniform','four_local','changed_tail'):
   rng=random.Random(771+m);routes=[rng.sample(range(32),8) for _ in range(m)]
   if pattern=='four_local':routes=[list(range(4))+rng.sample(range(4,32),4) for _ in range(m)]
   ids=torch.tensor(routes,device='cuda',dtype=torch.int32);ids[ids>=e]=-1
   if pattern=='changed_tail':ids[m//2:]=-1
   active=ids.flatten()>=0;valid=ids.numel();cfg=next(c for lo,hi,c in baseline.fp32_configs['w13'] if lo<valid<=hi)
   aligned=moe_align_block_size(ids,cfg['block_shape'][0],e,ignore_invalid_expert=True)
   x=torch.randn((m,h),device='cuda',dtype=torch.bfloat16)/20
   outputs=[torch.empty((m*8,2*h),device='cuda',dtype=torch.bfloat16) for _ in range(2)]
   def call(layer,target):
    return HummingMethod.forward_layer(layer,inputs=x,outputs=target,top_k=8,sublayer_name='w13',tuning_config=layer.fp32_configs['w13'],sorted_ids=aligned[0],expert_ids=aligned[1],num_tokens_padded=aligned[2],valid_shape_m=valid,compute_config={'gemm_type':'indexed','use_f16_accum':False})
   graphs=[]
   for layer,target in zip((baseline,candidate),outputs):
    call(layer,target);call(layer,target);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):call(layer,target)
    graphs.append(graph)
   if args.fusion:
    import w13_fused_epilogue
    fused=w13_fused_epilogue.prepare(candidate,cfg)
    fused_output=torch.full((m*8,h),37.,device='cuda',dtype=torch.bfloat16)
    for _ in range(2):fused.run(candidate,x,fused_output,ids,aligned)
    torch.cuda.synchronize();fused_graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(fused_graph):fused.run(candidate,x,fused_output,ids,aligned)
    (out/('fused_m'+str(m)+'.cu')).write_text(fused.code)
    report.setdefault('fused_kernels',[]).append({'m':m,'configuration':cfg,'cubin':fused.kernel_filename,'source_sha256':hashlib.sha256(fused.code.encode()).hexdigest(),'dynamic_shared':fused.dynamic_shared})
   baseline_activation=torch.empty((m*8,h),device='cuda',dtype=torch.bfloat16)
   def baseline_component():
    call(baseline,outputs[0]);scale_active_rows(outputs[0],ids.flatten(),baseline.w13_post_scale)
    silu_and_mul(outputs[0],baseline_activation,expert_ids=ids.flatten(),expert_step=1)
   component_graph=capture(baseline_component)
   cfg2=next(c for lo,hi,c in baseline.fp32_configs['w2'] if lo<valid<=hi)
   aligned2=moe_align_block_size(ids,cfg2['block_shape'][0],e,ignore_invalid_expert=True)
   down_a=torch.empty((m*8,h),device='cuda',dtype=torch.bfloat16);down_b=torch.empty_like(down_a)
   scores=torch.rand((m,8),device='cuda');scores/=scores.sum(-1,keepdim=True)
   def down(layer,activation,target):
    HummingMethod.forward_layer(layer,inputs=activation,outputs=target,top_k=1,sublayer_name='w2',tuning_config=layer.fp32_configs['w2'],sorted_ids=aligned2[0],expert_ids=aligned2[1],num_tokens_padded=aligned2[2],valid_shape_m=valid,compute_config={'gemm_type':'indexed','use_f16_accum':False})
    scale_active_rows(target,ids.flatten(),layer.w2_post_scale)
    return fused_post_expert_norm_weighted_sum(target.view(m,8,h),scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=ids)
   for generation in range(3):
    torch.manual_seed(771+m+generation);x.copy_(torch.randn_like(x)/20)
    for graph in graphs:graph.replay()
    a=outputs[0][active];b=outputs[1][active][:,inverse]
    check(a,b,m=m,pattern=pattern,generation=generation,stage='raw_gemm')
    scales=baseline.w13_post_scale[ids.flatten()[active].long(),None]
    a=(a.float()*scales.float()).to(torch.bfloat16);b=(b.float()*scales.float()).to(torch.bfloat16)
    check(a,b,m=m,pattern=pattern,generation=generation,stage='scaled_bf16')
    oa=torch.empty((a.shape[0],h),device='cuda',dtype=torch.bfloat16);ob=torch.empty_like(oa)
    silu_and_mul(a,oa);silu_and_mul(b,ob)
    check(oa,ob,m=m,pattern=pattern,generation=generation,stage='stock_silu')
    if args.fusion:
     fused_graph.replay()
     assert torch.all(fused_output[~active]==37.), 'Inactive row overwritten'
     check(oa,fused_output[active],m=m,pattern=pattern,generation=generation,stage='fused_epilogue')
     component_graph.replay()
     check(baseline_activation[active],fused_output[active],m=m,pattern=pattern,generation=generation,stage='component_graph')
     final_a=down(baseline,baseline_activation,down_a);final_b=down(candidate,fused_output,down_b)
     check(down_a[active],down_b[active],m=m,pattern=pattern,generation=generation,stage='down_projection')
     check(final_a,final_b,m=m,pattern=pattern,generation=generation,stage='final_norm_weighted_sum')
   for cold in (False,True):
    timings={'A1':[],'A2':[],'B':[]}
    for repeat in range(12):
     sequence=[('A1',component_graph),('B',fused_graph),('A2',component_graph)]
     if repeat%2:sequence.reverse()
     for label,graph in sequence:timings[label].append(sample(graph,cold))
    med={key:statistics.median(values) for key,values in timings.items()};aa=max(med['A1'],med['A2'])/min(med['A1'],med['A2'])
    row={'m':m,'pattern':pattern,'cache_sweep':cold,'samples':timings,'median_us':med,'aa_ratio':aa,'aa_qualified':aa<=1.10,'component_speedup':statistics.median(timings['A1']+timings['A2'])/med['B'],'scope':'Local SM86 four-expert W13 GEMM plus scale and SiLU versus fused epilogue; not serving;64MiB cache sweep outside events is an attempt, not residency proof'}
    report.setdefault('timings',[]).append(row);save()
   del component_graph,baseline_activation,down_a,down_b,aligned2
   if args.fusion:del fused_graph,fused_output
   del graphs,outputs,x,ids,aligned
   torch.cuda.empty_cache();print(json.dumps({'m':m,'pattern':pattern,'checks':len(report['checks'])}),flush=True)
 report['status']='PASSED_LOCAL_FUSED_EPILOGUE' if args.fusion else 'PASSED_LOCAL_PERMUTATION_PRECONDITION';report['pending']=['Hopper numerical validation','Full-model integration and serving performance','One canonical representation startup peak measurement','Graph-buffer alias integration','Current standalone driver launcher is research-only'];save()
if __name__=='__main__':main()
