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
 p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');p.add_argument('--fusion',action='store_true');p.add_argument('--out',required=True);p.add_argument('--experts',type=int,default=32);p.add_argument('--tokens',nargs='+',type=int,default=[8,32,128,768,1024]);p.add_argument('--patterns',nargs='+',default=['uniform','hot_rank','tail_invalid','all_remote']);args=p.parse_args();out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
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
 print('fixture_ready',flush=True)


 HOPPER=OWN.parent/'hopper_moe';sys.path.insert(0,str(HOPPER))
 from virtual_k_codegen import render_virtual_k
 from configuration_contract import m2_compatible
 from humming.kernel.humming import HummingKernel
 from humming import ops
 from sglang.kernels.jit.utils import load_jit,make_cpp_args,is_arch_support_pdl,get_activation_cuda_cflags
 from interleaved_dispatch import activate
 cpp=make_cpp_args(torch.bfloat16,is_arch_support_pdl())
 module=load_jit('research_interleaved_activation',*cpp,cuda_files=[str(OWN/'interleaved_activation.cuh')],extra_cuda_cflags=get_activation_cuda_cflags(),cuda_wrappers=[('run_activation_filtered',f'InterleavedActivationKernel<{cpp}>::run_activation_filtered')])
 jobs=json.loads((HOPPER/'LEGAL_M32_JOBS.json').read_text())['jobs'];configs={j['label']:j['configuration'] for j in jobs}
 config=configs['m2'];enabled=True
 class ResearchKernel(HummingKernel):
  def __new__(cls,*a,**kw):
   obj=object.__new__(cls);obj.inited=False;obj.cubin_loaded=False;return obj
  def prepare(self):
   assert self.sm_version==86 and not self.use_tma and not self.use_warp_spec
   self.code,self.research=render_virtual_k(self.code,config,86,enabled)
   super().prepare()
 kernel=ResearchKernel(**config);table=torch.tensor([0,1<<30,kernel.kernel_id,0],device='cpu',dtype=torch.int64)
 locks=torch.zeros(2048,device='cuda',dtype=torch.int32)
 assert m2_compatible(configs['heavy'],config,16,sm_version=86)
 assert not m2_compatible(configs['heavy'],config,16,sm_version=90)
 assert not m2_compatible(configs['heavy'],config,8,sm_version=86)
 plan=split_module.choose_partition_configs(candidate,256,256)
 for name in ('light','heavy'):
  for key,value in plan[name]['w13'].items():
   if key in configs[name]:assert (list(value) if isinstance(value,tuple) else value)==configs[name][key],(name,key,value,configs[name][key])
 assert plan['cutoff']==16
 # Preserve actual helper statement order, aliasing, scaling and W2 verbatim.
 tree=ast.parse(split_path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='split_expert_outputs')
 code=ast.unparse(node)
 old="return original_helper(layer, x, local_ids, torch, retain=retain, defer_downscale=defer_downscale)"
 assert code.count(old)==1;code=code.replace(old,"raise ValueError('Unsupported research geometry; production requires preparation-time canonical fallback')")
 old="HummingMethod.forward_layer(layer, inputs=x, outputs=cache1, top_k=topk, sublayer_name='w13', tuning_config=config, sorted_ids=sorted_ids, expert_ids=expert_ids, num_tokens_padded=padded, valid_shape_m=valid, compute_config={'gemm_type': 'indexed', 'use_f16_accum': False})"
 assert code.count(old)==1
 code=code.replace(old,"research_forward(name, ids, alignment, layer, x, cache1, topk, config, sorted_ids, expert_ids, padded, valid)")
 old='silu_and_mul(cache1, cache2, expert_ids=local_ids.flatten(), expert_step=1)';assert code.count(old)==1
 code=code.replace(old,"research_activation(cache1, cache2, local_ids.flatten())")
 old='return (cache3.view(m, topk, k), local_ids)';assert code.count(old)==1
 code=code.replace(old,'return (cache3.view(m, topk, k), local_ids, cache2)')
 (out/'candidate_extracted.py').write_text(code+'\n')
 dispatch={'m2':True,'calls':[]}
 def research_forward(name,local,alignment,layer,x,output,topk,cfg,sorted_ids,expert_ids,padded,valid):
  if name=='heavy' and dispatch['m2']:
   sorted_ids,expert_ids,padded=alignment(name,local,32)
   ops.launch_kernel(configs=table,inputs=x,weight=layer.w13_weight,weight_scale=layer.w13_weight_scale,outputs=output,sorted_ids=sorted_ids,expert_ids=expert_ids,num_tokens_padded=padded,locks=locks,top_k=topk,valid_shape_m=valid)
  else:HummingMethod.forward_layer(layer,inputs=x,outputs=output,top_k=topk,sublayer_name='w13',tuning_config=cfg,sorted_ids=sorted_ids,expert_ids=expert_ids,num_tokens_padded=padded,valid_shape_m=valid,compute_config={'gemm_type':'indexed','use_f16_accum':False})
 def research_activation(x,out,ids):
  activate('gate_up_interleaved',x,out,ids,lambda x,o,i:silu_and_mul(x,o,expert_ids=i,expert_step=1),lambda x,o,i:module.run_activation_filtered(x,o,i,1,'silu'))
 space=dict(vars(split_module),research_forward=research_forward,research_activation=research_activation);exec(compile(code,str(out/'candidate_extracted.py'),'exec'),space)
 modified=space['split_expert_outputs']
 import sglang.kernels.ops.activation.activation as actmod
 native_activation=actmod.silu_and_mul;observed=[]
 def observing_activation(x,out,*a,**kw):
  result=native_activation(x,out,*a,**kw);observed.append(out);return result
 m=32;ids=torch.empty((m,8),device='cuda',dtype=torch.int32);x=torch.empty((m,h),device='cuda',dtype=torch.bfloat16);scores=torch.empty((m,8),device='cuda')
 report={'status':'RUNNING','scope':'SM86 synthetic E32 H4096 M32 full scored helper plus finalizer; one interleaved candidate shared by M2 and fallback graphs. No serving/H200 claim.','plan':plan,'checks':[],'routes':[],'timings':[]}
 def save():(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
 def check(label,a,b):
  equal=torch.equal(a.contiguous().view(torch.int16),b.contiguous().view(torch.int16));finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all());report['checks'].append({'label':label,'bitwise':equal,'finite':finite});save();assert equal and finite,label
 def setcase(pattern,generation):
  rng=random.Random(980+generation);global_ids=[]
  for token in range(m):
   if pattern=='uniform':row=rng.sample(range(256),8)
   elif pattern=='hot_one':row=[generation%3]+rng.sample(range(32,256),7)
   elif pattern=='hot_eight':row=rng.sample(range(8),8)
   elif pattern=='crossing':
    limits=(15,17) if generation%2==0 else (17,15)
    row=[expert for expert,limit in enumerate(limits) if token<limit]
    row+=rng.sample(range(32,256),8-len(row))
   elif pattern=='all_remote':row=rng.sample(range(32,256),8)
   else:raise ValueError(pattern)
   assert len(set(row))==8;global_ids.append(row)
  local=[[v if v<32 else -1 for v in row] for row in global_ids]
  ids.copy_(torch.tensor(local,device='cuda',dtype=torch.int32));torch.manual_seed(390+generation);x.copy_(torch.randn_like(x)/20);scores.copy_(torch.rand_like(scores));scores.div_(scores.sum(-1,keepdim=True))
  report['routes'].append({'pattern':pattern,'generation':generation,'unique_global_top8':True,'counts':[sum(v==e for row in local for v in row) for e in range(32)]});save()
 def finish(value,local):return fused_post_expert_norm_weighted_sum(value,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=local,expert_output_scale=baseline.w2_post_scale)
 def run(name):
  if name=='original':
   actmod.silu_and_mul=observing_activation
   try:value,local=ns['_humming_expert_outputs'](baseline,x,ids,torch,defer_downscale=True,global_experts=256)
   finally:actmod.silu_and_mul=native_activation
   activation=observed[-1]
  else:
   dispatch['m2']=m2_compatible(configs['heavy'],config,16,sm_version=86 if name=='m2' else 90);value,local,activation=modified(None,candidate,x,ids,torch,global_experts=256,defer_downscale=True)
  return value,activation,finish(value,local)
 setcase('uniform',0);graphs={};outputs={}
 for name in ('original','m2','fallback'):
  print('warm_'+name,flush=True)
  for _ in range(2):run(name)
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):outputs[name]=run(name)
  graphs[name]=g
 print('graphs_ready',flush=True)
 for pattern in ('uniform','hot_one','hot_eight','crossing','all_remote'):
  for generation in range(3):
   setcase(pattern,generation)
   for name,g in graphs.items():
    outputs[name][0].fill_(37);outputs[name][1].fill_(37);g.replay()
   torch.cuda.synchronize();active=ids>=0;flat=active.flatten()
   for name in ('m2','fallback'):
    for stage,index,mask in (('activation',1,flat),('down',0,active)):
     check('%s_%s_%s_%d'%(name,stage,pattern,generation),outputs[name][index][mask],outputs['original'][index][mask])
    check('%s_final_%s_%d'%(name,pattern,generation),outputs[name][2],outputs['original'][2])
   for name in graphs:
    ok=bool((outputs[name][1][~flat]==37).all());report['checks'].append({'label':name+'_inactive_activation','passed':ok});assert ok
   save()
  print('passed_'+pattern,flush=True)
 report.update(status='PASS',peak_allocated=torch.cuda.max_memory_allocated(),check_count=len(report['checks']));save()
if __name__=='__main__':main()
