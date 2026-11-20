"""Prerequisites for an interleaved W13 scale/SiLU GEMM epilogue, not the fusion itself."""
import ast,hashlib,json,pathlib,sys,os,argparse,random,statistics,importlib.util,time
ROOT=pathlib.Path(__file__).resolve().parents[4]; OWN=pathlib.Path(__file__).resolve().parent
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
 p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');p.add_argument('--correctness-only',action='store_true');p.add_argument('--fusion',action='store_true');p.add_argument('--out',required=True);p.add_argument('--experts',type=int,default=32);p.add_argument('--tokens',nargs='+',type=int,default=[8,32,128,768,1024]);p.add_argument('--patterns',nargs='+',default=['uniform','hot_rank','tail_invalid','all_remote']);args=p.parse_args();out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
 report=cpu_contract();(out/'layout_contract.json').write_text(json.dumps(report,indent=2)+'\n')
 if args.gpu and not args.fusion:p.error('--gpu requires --fusion for this harness')
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
 sys.modules['sglang.kernels.ops.moe.muse_spark_humming_scale']=scale_module
 split_path=EXPORT/'python/sglang/kernels/ops/moe/muse_spark_humming_split.py'
 spec=importlib.util.spec_from_file_location('sglang.kernels.ops.moe.muse_spark_humming_split',split_path);split_module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=split_module;spec.loader.exec_module(split_module)
 finalizer_path=EXPORT/'python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py'
 spec=importlib.util.spec_from_file_location('round2_scored_finalizer',finalizer_path);finalizer_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(finalizer_module)
 fused_post_expert_norm_weighted_sum=finalizer_module.fused_post_expert_norm_weighted_sum
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
 tree=ast.parse(src.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_prepare_humming_experts','_humming_expert_outputs_default','_humming_expert_outputs')];fn=next(n for n in nodes if n.name=='_prepare_humming_experts');ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),ns)
 stock=ROOT/'sglang/python/sglang/test/test_marlin_utils.py';tree=ast.parse(stock.read_text());gen=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_nvfp4_weight_and_ref');gs={'torch':torch};exec(compile(ast.Module(body=[gen],type_ignores=[]),str(stock),'exec'),gs)
 torch.manual_seed(771);h=4096;e=args.experts;raw=torch.nn.Module()
 for prefix,n in (('w13',2*h),('w2',h)):
  weights=[];scales=[];globals_=[]
  for expert in range(e):
   w,s,g,ref=gs[gen.name](n,h,torch.bfloat16);weights.append(w);scales.append(s);globals_.append(g);del ref
  for suffix,values in (('weight',weights),('weight_scale',scales),('weight_scale_2',globals_)):
   setattr(raw,prefix+'_'+suffix,torch.nn.Parameter(torch.stack(values),requires_grad=False))
  del weights,scales,globals_,w,s,g,values
 for stage in ('w13','w2'):
  getattr(raw,stage+'_weight_scale_2').mul_(torch.linspace(.25,2.,e,device='cuda',dtype=torch.bfloat16))
 perm=torch.stack((torch.arange(h,device='cuda'),torch.arange(h,2*h,device='cuda')),dim=1).flatten();inverse=torch.argsort(perm)
 with cached_threshold_for_preparation(out):
  baseline=ns[fn.name](raw,torch)
  raw.w13_weight=torch.nn.Parameter(raw.w13_weight[:,perm].contiguous(),requires_grad=False)
  # Index bytes because native float8 advanced indexing is not universal.
  raw.w13_weight_scale=torch.nn.Parameter(raw.w13_weight_scale.view(torch.uint8)[:,perm].contiguous().view(torch.float8_e4m3fn),requires_grad=False)
  candidate=ns[fn.name](raw,torch)
 del raw
 assert baseline.fp32_configs==candidate.fp32_configs

 from composed import Composed, scalable_partition
 report={'status':'RUNNING','scope':'Local SM86 full MoE helper plus finalizer; synthetic E32 H4096, no serving/H200 claim','checks':[],'timings':[],'sources':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (pathlib.Path(__file__),OWN/'composed.py',scalable_partition.__file__ and pathlib.Path(scalable_partition.__file__))}}
 def save():(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
 def check(a,b,**meta):
  finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all());equal=torch.equal(a.contiguous().view(torch.int16),b.contiguous().view(torch.int16))
  row={**meta,'finite':finite,'bitwise':equal,'max_abs':float((a.float()-b.float()).abs().max()) if a.numel() else 0.};report['checks'].append(row);save()
  if not finite or not equal:report['status']='FAILED_STRICT_GATE';save();raise AssertionError(row)
 for m in args.tokens:
  # Small decode is a scored-policy control, not the held small-K repair.
  fused=Composed(candidate,m*8,e*8,split=m*8<=4096)
  composed=Composed(candidate,m*8,e*8,split=True)
  report.setdefault('policies',[]).append({'m':m,'fused':fused.plan,'composed':composed.plan})
  for pattern in args.patterns:
   torch.manual_seed(1921+m);x=torch.randn((m,h),device='cuda',dtype=torch.bfloat16)/20
   ids=torch.empty((m,8),device='cuda',dtype=torch.int32)
   scores=torch.rand((m,8),device='cuda');scores/=scores.sum(-1,keepdim=True)
   def set_inputs(generation):
    rng=random.Random(1641+m+generation);routes=[]
    for token in range(m):
     pool=range(e) if pattern=='hot_rank' else range(e*8)
     if pattern=='all_remote':pool=range(e,e*8)
     row=rng.sample(pool,8)
     if pattern=='tail_invalid' and token>=max(0,m//2-generation):row=[-1]*8
     routes.append([v if 0<=v<e else -1 for v in row])
    if pattern=='crossing':
     cutoff=composed.plan.get('cutoff',1)
     counts=((cutoff-1,cutoff+1,3),(cutoff+1,3,cutoff-1),(0,0,0))[generation%3]
     for token,row in enumerate(routes):
      row[:]=rng.sample(range(e,e*8),8)
      for expert,count in enumerate(counts):
       if token<count:row[expert]=expert
     assert all(len(set(row))==8 for row in routes)
     routes=[[v if 0<=v<e else -1 for v in row] for row in routes]
    ids.copy_(torch.tensor(routes,device='cuda',dtype=torch.int32))
    torch.manual_seed(2341+m+generation);x.copy_(torch.randn_like(x)/20)
   set_inputs(0)
   def original():return ns['_humming_expert_outputs'](baseline,x,ids,torch,defer_downscale=True,global_experts=e*8)
   def split():return scalable_partition.split_expert_outputs_scalable(ns['_humming_expert_outputs_default'],baseline,x,ids,torch,defer_downscale=True,global_experts=e*8)
   raw={'original':original,'split':split,'fused':lambda:fused(x,ids),'composed':lambda:composed(x,ids)}
   def finish(value,local):return fused_post_expert_norm_weighted_sum(value,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=local,expert_output_scale=baseline.w2_post_scale)
   def call(name):return finish(*raw[name]())
   graphs={};outputs={}
   for name in raw:
    for _ in range(2):call(name)
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):value,local=raw[name]();final=finish(value,local)
    graphs[name]=g;outputs[name]=(value,final)
   for generation in range(3):
    set_inputs(generation)
    for value,final in outputs.values():value.fill_(37.)
    for graph in graphs.values():graph.replay()
    torch.cuda.synchronize();active=ids>=0
    for name in ('split','fused','composed'):
     check(outputs[name][0][active],outputs['original'][0][active],m=m,pattern=pattern,generation=generation,variant=name,stage='raw_down_graph')
     check(outputs[name][1],outputs['original'][1],m=m,pattern=pattern,generation=generation,variant=name,stage='final_graph')
   # Full captured helper includes dynamic classification, alignment, both GEMMs and finalizer.
   set_inputs(0)
   for cold in (() if args.correctness_only else (False,True)):
    warm=time.monotonic()
    while time.monotonic()-warm<.25:
     for graph in graphs.values():graph.replay()
     torch.cuda.synchronize()
    for target in ('split','fused','composed'):
     timings={k:[] for k in ('A1','B1','A2','B2')}
     for phase in (1,2):
      for rep in range(8):
       seq=[('A',graphs['original']),('B',graphs[target])]
       if rep%2:seq.reverse()
       for key,graph in seq:timings[key+str(phase)].append(sample(graph,cold))
     med={key:statistics.median(v) for key,v in timings.items()};controls={key:med[key+'1']/med[key+'2'] for key in ('A','B')}
     report['timings'].append({'m':m,'pattern':pattern,'variant':target,'cache_sweep':cold,'samples_us':timings,'medians_us':med,'controls':controls,'qualified':all(.9<=v<=1.1 for v in controls.values()),'ratio':statistics.median(timings['A1']+timings['A2'])/statistics.median(timings['B1']+timings['B2'])});save()
   print(json.dumps({'m':m,'pattern':pattern,'checks':len(report['checks'])}),flush=True)
   del graphs,outputs,g,value,final,x,ids,scores,active
   torch.cuda.empty_cache()
 report['status']='COMPLETED_LOCAL_FOUR_WAY';report['peak_allocated']=torch.cuda.max_memory_allocated();save()
if __name__=='__main__':main()
