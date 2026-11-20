"""Pinned Humming 0.1.12 adapter experiment with stock BF16 scale boundaries."""
from __future__ import annotations
import argparse,ast,hashlib,json,sys,time,importlib.util,statistics,copy,os,random
from pathlib import Path
from types import MethodType,SimpleNamespace
OWN=Path(__file__).resolve().parent;ROOT=OWN.parents[2];SWARM=ROOT/'swarm_2026-11-16';HERE=SWARM/'moe'
sys.path.insert(0,str(HERE/'phase2'))
sys.path.insert(0,str(HERE))
COMBINED=HERE.parent/'coordination/round5_humming_adaptive_fused/candidate_sources'

sys.path.insert(0,str(HERE/'humming_vendor'))
sys.path.insert(0,str(ROOT/'optimization_marlin_tiles_2026-11-15'))

from fixture_tuning_cache import cached_threshold_for_preparation, validate_all_owned_intervals

def main():
    p=argparse.ArgumentParser();p.add_argument('--sglang-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--hidden',type=int,default=4096);p.add_argument('--experts',type=int,default=32);p.add_argument('--tokens',nargs='+',type=int,default=[753,3012])
    p.add_argument('--candidate',choices=['scalable_partition.py','unsplit_partition.py','oneshot_partition.py','fused_partition.py'],default='scalable_partition.py');p.add_argument('--patterns',nargs='+',default=['uniform','hot_expert','all_remote']);p.add_argument('--original-policy',action='store_true');p.add_argument('--correctness-only',action='store_true');p.add_argument('--samples',type=int,default=12);p.add_argument('--graph-body',type=int,default=1);p.add_argument('--id-dtype',choices=['int32','int64'],default='int32');p.add_argument('--different-blocks',action='store_true');p.add_argument('--import-only',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    milestones=[]
    def mark(stage,**details):
        item={'stage':stage,'unix_seconds':time.time(),**details}
        torch_module=sys.modules.get('torch')
        if torch_module is not None and torch_module.cuda.is_initialized():
            free,total=torch_module.cuda.mem_get_info()
            item['memory_bytes']={'allocated':torch_module.cuda.memory_allocated(),'reserved':torch_module.cuda.memory_reserved(),'peak_allocated':torch_module.cuda.max_memory_allocated(),'peak_reserved':torch_module.cuda.max_memory_reserved(),'device_free':free,'device_total':total}
        milestones.append(item)
        (args.out/'milestones.json').write_text(json.dumps(milestones,indent=2)+'\n')
        print(json.dumps(item),flush=True)
    mark('main_entered')
    if os.environ.get('MAX_JOBS')!='1' or os.environ.get('HUMMING_DISABLE_PARALLEL_BUILD')!='1':raise RuntimeError('Serial compiler environment required')
    if args.different_blocks:raise ValueError('EP experiment cannot alter independent block policy')
    assert hashlib.sha256((COMBINED.parent/'candidate.diff').read_bytes()).hexdigest()=='8b3ed81e053e4ccb2a043afd25cb94bb49017995ece517bb35aeb6e1b79e026b'
    manifest=json.loads((HERE.parent/'projection_communication/adaptive_split_v3/manifest.json').read_text())
    pinned={COMBINED/rel:digest for rel,digest in manifest['sources'].items()}
    reference_finalizer=args.sglang_root/'python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py'
    pinned[reference_finalizer]='d87fcf2d67a6ae92d3b2e08d9dfa670dbdb7663c4683a6cb933d5c3b8cf42fe8'
    for path,digest in pinned.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,(path,'changed source')
    from probe_preflight import preflight
    provenance=preflight(args.sglang_root)
    if not args.tokens or any(m<=0 for m in args.tokens):
        raise ValueError('Provide positive token sizes; the serving parent handles zero tokens')
    from kernel_runtime import bootstrap
    runtime=bootstrap(args.sglang_root)
    import torch
    from humming.layer import HummingMethod
    import humming
    if args.import_only:
        print(json.dumps({'humming_file':humming.__file__,'torch':torch.__version__,'cuda':torch.version.cuda}),flush=True);return
    mark('runtime_imports_ready')
    from marlin_bench import load_method,event_sample
    if args.graph_body != 1:raise ValueError('Stage profiling requires graph body exactly 1')
    args.correctness_only=True
    if args.samples<6:raise ValueError('At least six samples required')
    if args.graph_body>8 and args.hidden>=4096:raise ValueError('Large-geometry pilot limits graph-body memory to 8')
    id_dtype=getattr(torch,args.id_dtype)
    def capture(torch,call):
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):call()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.graph_body):output=call()
        return graph,output
    from tile_study import make_routes as stock_make_routes
    def make_routes(m,topk,experts,ep,pattern,seed):
        if pattern=='tail_invalid':
            rows=stock_make_routes(m,topk,experts,ep,'uniform',seed)
            for row in rows:row[-3:]=[-1]*3
            rows[-1]=[-1]*topk
            return rows
        if pattern=='all_local_uniform':
            rng=random.Random(seed);return [rng.sample(range(experts//ep),topk) for _ in range(m)]
        if pattern=='duplicates':return [[0]*topk for _ in range(m)]
        if pattern=='mixed_duplicate':return [[0 if i%2==0 else experts-1]*topk for i in range(m)]
        if pattern!='mixed':return stock_make_routes(m,topk,experts,ep,pattern,seed)
        rng=random.Random(seed)
        return [rng.sample(range(topk),topk) if i<m//2 else rng.sample(range(experts),topk) for i in range(m)]
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
    from sglang.kernels.ops.moe.muse_spark_post_expert_finalize_triton import fused_post_expert_norm_weighted_sum
    tree=ast.parse((args.sglang_root/'python/sglang/test/test_marlin_utils.py').read_text())
    f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_nvfp4_weight_and_ref')
    ns={'torch':torch};exec(compile(ast.Module(body=[f],type_ignores=[]),'<stock-generator>','exec'),ns)
    mark('raw_weights_begin')
    torch.manual_seed(216);raw=torch.nn.Module()
    raw.quant_config=SimpleNamespace(group_size=16);raw.moe_runner_config=SimpleNamespace(is_gated=True)
    raw.params_dtype=torch.bfloat16;raw.intermediate_size_per_partition=args.hidden
    raw.num_local_experts=args.experts;raw.global_num_experts=args.experts*8
    raw._expert_map=torch.full((args.experts*8,),-1,device='cuda',dtype=torch.int32);raw._expert_map[:args.experts]=torch.arange(args.experts,device='cuda',dtype=torch.int32)
    raw._expert_lookup=torch.cat((raw._expert_map.new_full((1,),-1),raw._expert_map))
    for prefix,n in (('w13',args.hidden*2),('w2',args.hidden)):
        packed=[];scales=[];gs=[]
        for e in range(args.experts):
            w,s,g,ref=ns[f.name](n,args.hidden,torch.bfloat16);packed.append(w);scales.append(s);gs.append(g);del ref
        for suffix,vs in (('weight',packed),('weight_scale',scales),('weight_scale_2',gs)):
            setattr(raw,prefix+'_'+suffix,torch.nn.Parameter(torch.stack(vs),requires_grad=False))
    # Release generator-owned stage lists before preparing either canonical layout.
    del packed,scales,gs,w,s,g,vs
    mark('raw_weights_complete_helper_import_begin')
    candidate_root=COMBINED
    candidate_export=candidate_root/'python/sglang/srt/layers/muse_spark/muse_spark_nvfp4_moe.py'
    candidate_kernel=candidate_root/'python/sglang/kernels/ops/moe/muse_spark_humming_scale.py'
    kernel_name='sglang.kernels.ops.moe.muse_spark_humming_scale'
    spec=importlib.util.spec_from_file_location(kernel_name,candidate_kernel)
    module=importlib.util.module_from_spec(spec);sys.modules[kernel_name]=module;spec.loader.exec_module(module)
    split_path=candidate_root/'python/sglang/kernels/ops/moe/muse_spark_humming_split.py'
    split_name='sglang.kernels.ops.moe.muse_spark_humming_split'
    split_spec=importlib.util.spec_from_file_location(split_name,split_path)
    split_module=importlib.util.module_from_spec(split_spec);sys.modules[split_name]=split_module;split_spec.loader.exec_module(split_module)
    choose_partition_configs=split_module.choose_partition_configs;partition_routes=split_module.partition_routes
    finalizer_path=candidate_root/'python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py'
    finalizer_spec=importlib.util.spec_from_file_location('combined_actual_finalizer',finalizer_path)
    finalizer_module=importlib.util.module_from_spec(finalizer_spec);sys.modules[finalizer_spec.name]=finalizer_module;finalizer_spec.loader.exec_module(finalizer_module)
    candidate_tree=ast.parse(candidate_export.read_text(encoding='utf-8'))
    candidate_nodes=[n for n in candidate_tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_prepare_humming_experts','_humming_expert_outputs_default','_humming_expert_outputs')]
    helper_text=ast.unparse(ast.Module(body=candidate_nodes,type_ignores=[]))
    helper_ns={'torch':torch}
    exec(compile(ast.Module(body=candidate_nodes,type_ignores=[]),str(candidate_export),'exec'),helper_ns)
    # Metadata guards do not read route values or synchronize the device.
    module.scale_active_rows(torch.empty((0,args.hidden),device='cuda',dtype=torch.bfloat16),torch.empty((0,),device='cuda',dtype=torch.int32),torch.ones((args.experts,),device='cuda',dtype=torch.bfloat16))
    (args.out/'executed_helper.py').write_text(helper_text)


    mark('helper_import_complete')
    norm=torch.linspace(-.5,.5,args.hidden,device='cuda',dtype=torch.bfloat16)
    weights={'post_norm_w':norm}
    for prefix in ('w13','w2'):
        weights[prefix]=getattr(raw,prefix+'_weight')
        weights[prefix+'_sf']=getattr(raw,prefix+'_weight_scale')
        gs=getattr(raw,prefix+'_weight_scale_2')
        gs.mul_(torch.linspace(.25,2.0,args.experts,device='cuda'))
        weights[prefix+'_gs']=gs
    with cached_threshold_for_preparation(args.out):
        mark('canonical_prepare_begin')
        state={'layer':helper_ns['_prepare_humming_experts'](raw,torch),'norm_w':(norm.float()+1).to(torch.bfloat16)}
        del weights
        humm=state['layer']
        mark('canonical_prepare_complete_grouped_configs_begin')
        grouped_configs={}
        mark('grouped_configs_complete')
        (args.out/'owned_configs.json').write_text(json.dumps({'indexed':humm.fp32_configs,'grouped':grouped_configs,'meta':{k:json.loads(v.to_str()) for k,v in humm.humming_metas.items()},'device':torch.cuda.get_device_name(),'sm_count':torch.cuda.get_device_properties(0).multi_processor_count,'source_harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    mark('uncached_config_validation_begin')
    config_equality=validate_all_owned_intervals(humm,grouped_configs,args.out)
    mark('uncached_config_validation_complete',intervals=config_equality['intervals_checked'])


    lookup=raw._expert_lookup;del raw
    norm=state['norm_w'];import collections
    spec=importlib.util.spec_from_file_location('phase4_scalable_partition',OWN/args.candidate)
    scalable=importlib.util.module_from_spec(spec);sys.modules[spec.name]=scalable;spec.loader.exec_module(scalable)
    if args.original_policy:scalable.choose_partition_configs=scalable.original_choose_partition_configs
    report={'status':'running','scope':'Numerical-only full corrected helper versus scored helper under external sanitizer; fresh allocations, poisoned reuse attempt, changed graph routes. No performance or H200 claim. Poison reuse is allocator-dependent and does not itself establish initcheck coverage.','source_sha256':{str(path):digest for path,digest in pinned.items()},'candidate_sha256':hashlib.sha256((OWN/args.candidate).read_bytes()).hexdigest(),'geometry':{'E':args.experts,'global_E':args.experts*8,'H':args.hidden,'topk':8},'metadata':[],'fullchain':[]}
    def save():
      t=args.out/'results.tmp';t.write_text(json.dumps(report,indent=2)+'\n');t.replace(args.out/'results.json')
    save()
    def cpu_ids(n,kind,seed):
      rng=random.Random(seed)
      if kind=='all_remote':return [-1]*n
      if kind=='hot':return [0]*n
      if kind=='balanced':return [i%args.experts for i in range(n)]
      if kind=='invalid':return [(-2,-1,0,args.experts-1,args.experts)[i%5] for i in range(n)]
      return [rng.randrange(-1,args.experts) for _ in range(n)]
    def metadata_check(tensor,values,result,context):
      hist=collections.Counter(v for v in values if 0<=v<args.experts);cutoff=40
      gold=([v if 0<=v<args.experts and hist[v]<=cutoff else -1 for v in values],[v if 0<=v<args.experts and hist[v]>cutoff else -1 for v in values],[hist[e] for e in range(args.experts)])
      flags=[torch.equal(actual,torch.tensor(expected,device='cuda',dtype=torch.int32).reshape(actual.shape)) for actual,expected in zip(result,gold)]
      if len(values)<=4096:
        prior=split_module.partition_routes(tensor,args.experts,cutoff);flags += [torch.equal(a,b) for a,b in zip(result,prior)]
      row={**context,'checks':len(flags),'passed':all(flags)};report['metadata'].append(row);save();assert row['passed'],row
    # Metadata has separate memcheck/racecheck coverage; focus sanitizer time on full helper.
    for dtype in ():
      for n in (4096,4097,6024,24096,131072):
        for kind in ('all_remote','hot','balanced','invalid','random'):
          values=cpu_ids(n,kind,17)
          if kind=='invalid' and dtype==torch.int64:values[0]=2**35
          ids=torch.tensor(values,device='cuda',dtype=dtype)
          result=scalable.partition_routes_scalable(ids,args.experts,40)
          metadata_check(ids,values,result,{'routes':n,'dtype':str(dtype),'pattern':kind,'mode':'eager'})
        if n in (6024,131072):
          graph=torch.cuda.CUDAGraph()
          with torch.cuda.graph(graph):output=scalable.partition_routes_scalable(ids,args.experts,40)
          for kind in ('all_remote','hot','balanced'):
            values=cpu_ids(n,kind,29);ids.copy_(torch.tensor(values,device='cuda',dtype=dtype));graph.replay()
            metadata_check(ids,values,output,{'routes':n,'dtype':str(dtype),'pattern':kind,'mode':'changed_graph'})
          del graph,output
    mark('metadata_gate_complete',checks=sum(r['checks'] for r in report['metadata']))
    def compare(actual,expected,context):
      diff=actual.float()-expected.float()
      row={**context,'relative_l2':(diff.norm()/expected.float().norm().clamp_min(1e-20)).item(),'max_abs':diff.abs().max().item(),'bitwise':torch.equal(actual.view(torch.int16),expected.view(torch.int16)),'finite':bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())}
      row['passed']=row['finite'] and row['relative_l2']<=.003 and row['max_abs']<=.03125
      journal.append(row);(args.out/'comparison_journal.json').write_text(json.dumps(journal,indent=2)+'\n')
      if not row['passed']:report.update(status='failed_numerical_gate',failure=row);save();raise AssertionError(row)
      return row
    journal=[]
    def graph_one(call):
      for _ in range(2):call()
      torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
      with torch.cuda.graph(g):o=call()
      return g,o
    def sample(graph):
      a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();graph.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
    for m in args.tokens:
      plan=scalable.choose_partition_configs(humm,m*8,args.experts*8)
      if plan is None:report['fullchain'].append({'m':m,'status':'no_distinct_plan'});save();continue
      for pattern in args.patterns:
        # Fresh backing allocations before numerical warmup.
        torch.cuda.synchronize();torch.cuda.empty_cache()
        mark('fresh_allocator_case',m=m,pattern=pattern)
        torch.manual_seed(216+m);x=torch.randn((m,args.hidden),device='cuda',dtype=torch.bfloat16)/20
        ids=torch.tensor(make_routes(m,8,args.experts*8,8,pattern,216+m),device='cuda',dtype=id_dtype);local=lookup[ids+1]
        scores=torch.rand((m,8),device='cuda');scores/=scores.sum(-1,keepdim=True)
        def a_raw():return helper_ns['_humming_expert_outputs'](humm,x,local,torch,defer_downscale=True,global_experts=args.experts*8)
        def b_raw():return scalable.split_expert_outputs_scalable(helper_ns['_humming_expert_outputs_default'],humm,x,local,torch,defer_downscale=True,global_experts=args.experts*8)
        def finish(value,local_ids):return finalizer_module.fused_post_expert_norm_weighted_sum(value,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=local_ids,expert_output_scale=humm.w2_post_scale)
        def a_call():return finish(*a_raw())
        def b_call():return finish(*b_raw())
        mark('fullchain_case_begin',m=m,pattern=pattern)
        av,_=a_raw();bv,_=b_raw();active=(local>=0)&(local<args.experts)
        raw_check=compare(bv[active],av[active],{'m':m,'pattern':pattern,'stage':'raw_down'}) if bool(active.any()) else {'scope':'No valid raw slots'}
        del av,bv,active
        final_check=compare(b_call(),a_call(),{'m':m,'pattern':pattern,'stage':'final'})
        # Deliberately poison storage, then exercise allocator reuse independently.
        poison=[torch.empty((m,8,args.hidden),device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        for allocation in poison:allocation.fill_(float('nan'))
        torch.cuda.synchronize();del allocation,poison
        mark('poisoned_allocator_case',m=m,pattern=pattern)
        poison_check=compare(b_call(),a_call(),{'m':m,'pattern':pattern,'stage':'poisoned_final'})
        ga,oa=graph_one(a_call);gb,ob=graph_one(b_call)
        checks=[]
        for seed,mode in ((223,'uniform'),(227,'hot_expert'),(239,'all_remote'),(241,'tail_invalid')):
          torch.manual_seed(seed);x.copy_(torch.randn_like(x)/20);ids.copy_(torch.tensor(make_routes(m,8,args.experts*8,8,mode,seed),device='cuda',dtype=id_dtype));local.copy_(lookup[ids+1]);ga.replay();gb.replay();torch.cuda.synchronize()
          checks.append(compare(ob,oa,{'m':m,'pattern':pattern,'changed_mode':mode,'seed':seed,'stage':'graph_final'}))
        if args.correctness_only:
          report['fullchain'].append({'m':m,'pattern':pattern,'plan':plan,'raw_check':raw_check,'final_check':final_check,'poisoned_check':poison_check,'changed_graph_checks':checks,'timing':'disabled for sanitizer'})
          save();mark('fullchain_case_complete',m=m,pattern=pattern)
          del ga,gb,oa,ob,x,ids,local,scores
          continue
        # Restore the stated workload for balanced A/B timings.
        ids.copy_(torch.tensor(make_routes(m,8,args.experts*8,8,pattern,216+m),device='cuda',dtype=id_dtype));local.copy_(lookup[ids+1]);ga.replay();gb.replay();torch.cuda.synchronize()
        samples={name:[] for name in ('A1','B1','A2','B2')}
        for phase in (1,2):
          for i in range(args.samples):
            for name,g in ((('A',ga),('B',gb)) if i%2==0 else (('B',gb),('A',ga))):samples[name+str(phase)].append(sample(g))
        med={name:statistics.median(v) for name,v in samples.items()};controls={'A':med['A1']/med['A2'],'B':med['B1']/med['B2']}
        report['fullchain'].append({'m':m,'pattern':pattern,'plan':plan,'raw_check':raw_check,'final_check':final_check,'changed_graph_checks':checks,'samples_us':samples,'medians_us':med,'controls':controls,'noise_pass':all(.9<=v<=1.1 for v in controls.values()),'speedup':statistics.median(samples['A1']+samples['A2'])/statistics.median(samples['B1']+samples['B2'])})
        save();mark('fullchain_case_complete',m=m,pattern=pattern)
        del ga,gb,oa,ob,x,ids,local,scores
    report['status']='completed';save();mark('completed')
if __name__=='__main__':main()





