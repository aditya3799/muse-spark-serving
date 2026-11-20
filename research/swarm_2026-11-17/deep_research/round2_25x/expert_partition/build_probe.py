from pathlib import Path
p=Path(__file__).resolve().parent
root=p.parents[3]
src=(root/'swarm_2026-11-17/deep_research/coordination/probe_w13_fusion_native_e32.py').read_text(encoding='utf-8-sig')
src=src[:src.index(" report={'route_generation'")]
src=src.replace('parents[3]; OWN','parents[4]; OWN')
src=src.replace("p.add_argument('--patterns',nargs='+',default=['uniform','four_local','changed_tail'])", "p.add_argument('--patterns',nargs='+',default=['uniform','hot_rank','tail_invalid','all_remote'])")
src=src.replace("scale_active_rows=scale_module.scale_active_rows", "scale_active_rows=scale_module.scale_active_rows\n sys.modules['sglang.kernels.ops.moe.muse_spark_humming_scale']=scale_module\n split_path=EXPORT/'python/sglang/kernels/ops/moe/muse_spark_humming_split.py'\n spec=importlib.util.spec_from_file_location('sglang.kernels.ops.moe.muse_spark_humming_split',split_path);split_module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=split_module;spec.loader.exec_module(split_module)")
# Stock helper AST is required for the scored baseline, preserving its <=4096 adaptive behavior.
src=src.replace("fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_prepare_humming_experts');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(src),'exec'),ns)", "nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_prepare_humming_experts','_humming_expert_outputs_default','_humming_expert_outputs')];fn=next(n for n in nodes if n.name=='_prepare_humming_experts');ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),ns)")
src=src.replace(' from sglang.kernels.ops.moe.muse_spark_post_expert_finalize_triton import fused_post_expert_norm_weighted_sum', " finalizer_path=EXPORT/'python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py'\n spec=importlib.util.spec_from_file_location('round2_scored_finalizer',finalizer_path);finalizer_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(finalizer_module)\n fused_post_expert_norm_weighted_sum=finalizer_module.fused_post_expert_norm_weighted_sum")
src+='''
 from composed import Composed, scalable_partition
 report={'status':'RUNNING','scope':'Local SM86 full MoE helper plus finalizer; synthetic E32 H4096, no serving/H200 claim','checks':[],'timings':[],'sources':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (pathlib.Path(__file__),OWN/'composed.py',scalable_partition.__file__ and pathlib.Path(scalable_partition.__file__))}}
 def save():(out/'results.json').write_text(json.dumps(report,indent=2)+'\\n')
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
    for graph in graphs.values():graph.replay()
    torch.cuda.synchronize();active=ids>=0
    for name in ('split','fused','composed'):
     check(outputs[name][0][active],outputs['original'][0][active],m=m,pattern=pattern,generation=generation,variant=name,stage='raw_down_graph')
     check(outputs[name][1],outputs['original'][1],m=m,pattern=pattern,generation=generation,variant=name,stage='final_graph')
   # Full captured helper includes dynamic classification, alignment, both GEMMs and finalizer.
   set_inputs(0)
   for cold in (False,True):
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
'''
(p/'probe_composed.py').write_text(src,encoding='utf-8')
import ast
for name in ('probe_composed.py','composed.py','cpu_contract.py'):ast.parse((p/name).read_text(encoding='utf-8-sig'))
print('Three isolated sources parse; no GPU imports executed')


