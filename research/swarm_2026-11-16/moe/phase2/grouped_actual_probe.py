"""Pinned Humming 0.1.12 adapter experiment with stock BF16 scale boundaries."""
from __future__ import annotations
import argparse,ast,hashlib,json,sys,time,importlib.util,statistics,copy,os,random
from pathlib import Path
from types import MethodType,SimpleNamespace
OWN=Path(__file__).resolve().parent;HERE=OWN.parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
COMBINED=HERE.parent/'coordination/round5_humming_adaptive_fused/candidate_sources'

sys.path.insert(0,str(HERE/'humming_vendor'))
sys.path.insert(0,str(ROOT/'optimization_marlin_tiles_2026-11-15'))

from compact_grouped import compact_grouped_outputs, prepare_grouped_configs
from fixture_tuning_cache import cached_threshold_for_preparation, validate_all_owned_intervals

def main():
    p=argparse.ArgumentParser();p.add_argument('--sglang-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--hidden',type=int,default=4096);p.add_argument('--experts',type=int,default=32);p.add_argument('--tokens',nargs='+',type=int,default=[33,128])
    p.add_argument('--samples',type=int,default=12);p.add_argument('--graph-body',type=int,default=8);p.add_argument('--id-dtype',choices=['int32','int64'],default='int32');p.add_argument('--different-blocks',action='store_true');p.add_argument('--import-only',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
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
    if args.graph_body not in (1,8,32):raise ValueError('Graph body must be 1, 8, or 32')
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
        grouped_configs=prepare_grouped_configs(humm)
        mark('grouped_configs_complete')
        (args.out/'owned_configs.json').write_text(json.dumps({'indexed':humm.fp32_configs,'grouped':grouped_configs,'meta':{k:json.loads(v.to_str()) for k,v in humm.humming_metas.items()},'device':torch.cuda.get_device_name(),'sm_count':torch.cuda.get_device_properties(0).multi_processor_count,'source_harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    mark('uncached_config_validation_begin')
    config_equality=validate_all_owned_intervals(humm,grouped_configs,args.out)
    mark('uncached_config_validation_complete',intervals=config_equality['intervals_checked'])
    original_up_scale=humm.w13_post_scale.clone()
    original_down_scale=humm.w2_post_scale.clone()
    lookup=raw._expert_lookup
    del raw
    norm=state['norm_w']
    if args.samples<12 or args.samples%2:raise ValueError('Even sample count >=12 required')
    defaults=copy.deepcopy(humm.fp32_configs)
    report={'status':'running','argv':sys.argv,'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidate_sha256':{f:hashlib.sha256((OWN/f).read_bytes()).hexdigest() for f in ('compact_grouped.py',)},'source_hashes':{str(p):h for p,h in pinned.items()},'gpu':torch.cuda.get_device_name(),'geometry':{'experts':args.experts,'global_experts':args.experts*8,'hidden':args.hidden,'topk':8,'id_dtype':args.id_dtype},'graph_body':args.graph_body,'timing_scope':'Repeated identical-input graph-body throughput calibration; not autoregressive decode latency','scope':'SM86 source-minimal exact scored helper A versus compact grouped-masked candidate B; shared canonical weights and unchanged BF16 arithmetic; no H200 speed claim','rows':[]}
    def save():(args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    flush=torch.empty(64*1024*1024,device='cuda',dtype=torch.uint8)
    def reject_fallback(*args,**kwargs):
        raise AssertionError('Grouped pilot unexpectedly took scored fallback')
    comparison_journal=[]
    comparison_context={}
    def check(value,reference):
        difference=value.float()-reference.float()
        relative=(difference.norm()/reference.float().norm().clamp_min(1e-20)).item()
        maximum=difference.abs().max().item()
        passed=bool(torch.isfinite(value).all()) and bool(torch.isfinite(reference).all()) and relative<=.003 and maximum<=.03125
        result={'relative_l2':relative,'max_abs':maximum,'bitwise_equal':bool(torch.equal(value.view(torch.int16),reference.view(torch.int16))),'passed':passed}
        # Persist evidence before asserting so failed or interrupted cases survive.
        flat=difference.reshape(-1)
        worst=int(flat.abs().argmax().item()) if flat.numel() else None
        record={'context':dict(comparison_context),'unix_seconds':time.time(),
                'monotonic_seconds':time.monotonic(),'comparison':result,
                'elements':value.numel(),'different_elements':int((value!=reference).sum().item()),
                'reference_abs_max':reference.float().abs().max().item() if reference.numel() else None,
                'value_abs_max':value.float().abs().max().item() if value.numel() else None,
                'worst_flat_index':worst,
                'worst_reference':reference.reshape(-1)[worst].float().item() if worst is not None else None,
                'worst_value':value.reshape(-1)[worst].float().item() if worst is not None else None}
        comparison_journal.append(record)
        temporary=args.out/'comparison_journal.tmp'
        temporary.write_text(json.dumps(comparison_journal,indent=2)+'\n')
        temporary.replace(args.out/'comparison_journal.json')
        if not passed:
            report.update(status='failed_numerical_gate',failure=record);save()
        assert passed,result
        return result
    for m in args.tokens:
      selected_grouped={stage:next(config for lo,hi,config in intervals if lo<m*8<=hi) for stage,intervals in grouped_configs.items()}
      up=selected_grouped['w13']['block_shape'][0];down=selected_grouped['w2']['block_shape'][0];routes=m*8
      capacity=max(((routes+up-1)//up)*up,((routes+down-1)//down)*down)
      packed_rows=args.experts*capacity
      workspace=2*packed_rows*3*args.hidden+2*routes*3*args.hidden+4*packed_rows+4*routes+4*args.experts+8*1024*1024
      assert workspace<=512*1024*1024,('Expected active grouped path, budget would fallback',workspace)
      plan=choose_partition_configs(humm,m*8,args.experts*8)
      for pattern in ('uniform','all_local_uniform','duplicates','mixed_duplicate','all_remote'):
        x=torch.randn((m,args.hidden),device='cuda',dtype=torch.bfloat16)/20
        ids=torch.tensor(make_routes(m,8,args.experts*8,8,pattern,216),device='cuda',dtype=id_dtype)
        scores=torch.rand((m,8),device='cuda');scores/=scores.sum(-1,keepdim=True)
        def finish(value,local):return finalizer_module.fused_post_expert_norm_weighted_sum(value,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=local,expert_output_scale=humm.w2_post_scale)
        def scored():return finish(*helper_ns['_humming_expert_outputs'](humm,x,lookup[ids+1],torch,defer_downscale=True,global_experts=args.experts*8))
        def candidate():return finish(*compact_grouped_outputs(reject_fallback,humm,grouped_configs,x,lookup[ids+1],torch,args.experts*8,defer_downscale=True))
        mark('scored_first_call_begin',m=m,pattern=pattern)
        av=scored();torch.cuda.synchronize();mark('scored_first_call_complete',m=m,pattern=pattern)
        bv=candidate();torch.cuda.synchronize();mark('candidate_first_call_complete',m=m,pattern=pattern)
        comparison_context={'stage':'eager_final','m':m,'pattern':pattern}
        eager=check(bv,av)
        row={'m':m,'pattern':pattern,'selected_plan':plan,'eager':eager,'grouped_configs':selected_grouped,'capacity':capacity,'workspace_bound':workspace,'replays':[]}
        local=lookup[ids+1]
        mark('raw_compare_begin',m=m,pattern=pattern)
        raw_a,_=helper_ns['_humming_expert_outputs'](humm,x,local,torch,defer_downscale=True,global_experts=args.experts*8)
        raw_b,_=compact_grouped_outputs(reject_fallback,humm,grouped_configs,x,local,torch,args.experts*8,defer_downscale=True)
        active=(local>=0)&(local<args.experts)
        active_count=int(active.sum().item())
        row['raw_down_active_slots']=active_count
        comparison_context={'stage':'raw_down_active','m':m,'pattern':pattern}
        row['raw_down_comparison']=check(raw_b[active],raw_a[active]) if active_count else {'scope':'No active slots; outputs intentionally undefined before masked finalizer'}
        del raw_a,raw_b,active,local
        mark('raw_compare_complete',m=m,pattern=pattern)
        del av,bv
        mark('graph_capture_begin',m=m,pattern=pattern)
        ga,oa=capture(torch,scored);gb,ob=capture(torch,candidate)
        mark('graph_capture_complete',m=m,pattern=pattern)
        for seed,mode in ((221,'uniform'),(227,'duplicates'),(239,'all_remote'),(257,'all_local_uniform'),(251,pattern)):
          torch.manual_seed(seed);x.copy_(torch.randn_like(x)/20);ids.copy_(torch.tensor(make_routes(m,8,args.experts*8,8,mode,seed),device='cuda',dtype=id_dtype));scores.copy_(torch.rand_like(scores));scores.div_(scores.sum(-1,keepdim=True))
          humm.w13_post_scale.copy_(original_up_scale*(1.5 if seed==227 else 1));humm.w2_post_scale.copy_(original_down_scale*(.5 if seed==239 else 1))
          ga.replay();gb.replay();torch.cuda.synchronize()
          comparison_context={'stage':'graph_final','m':m,'pattern':pattern,'seed':seed,'mode':mode}
          comparison=check(ob,oa)
          assert bool(torch.isfinite(oa).all()) and bool(torch.isfinite(ob).all())
          if mode=='all_remote':assert bool((ob==0).all())
          row['replays'].append({'seed':seed,'mode':mode,'comparison':comparison,'changed_global_scales':True})
        humm.w13_post_scale.copy_(original_up_scale);humm.w2_post_scale.copy_(original_down_scale)
        samples={k:[] for k in ('A1','B1','A2','B2')}
        for order in (list(samples),list(reversed(samples))):
          for label in order:flush.zero_();event_sample(torch,(ga if label[0]=='A' else gb).replay,1)
        for rep in range(args.samples):
          for label in (list(samples) if rep%2==0 else list(reversed(samples))):flush.zero_();samples[label].append(event_sample(torch,(ga if label[0]=='A' else gb).replay,1)/args.graph_body)
        med={k:statistics.median(v) for k,v in samples.items()};arms={k:statistics.median(samples[k+'1']+samples[k+'2']) for k in 'AB'}
        row.update(samples_us=samples,medians_us=arms,speedup=arms['A']/arms['B'],controls={k:med[k+'1']/med[k+'2'] for k in 'AB'},noise_pass=all(abs(med[k+'1']/med[k+'2']-1)<=.1 for k in 'AB'),passed=True)
        report['rows'].append(row);save();print(json.dumps({k:v for k,v in row.items() if k!='samples_us'}),flush=True)
        del ga,gb,oa,ob;torch.cuda.synchronize();assert humm.fp32_configs==defaults
        mark('case_complete',m=m,pattern=pattern)
        if (args.out/'STOP_AFTER_CASE').exists():report['status']='stopped_cooperatively';save();return
    report.update(status='completed',passed=True);save()

if __name__=='__main__':main()
