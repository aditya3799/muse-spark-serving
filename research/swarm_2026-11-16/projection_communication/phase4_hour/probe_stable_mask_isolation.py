"""DIAGNOSTIC_NOT_PROMOTABLE: isolate fixed-policy mask invariance.
Original baseline-qualification rejection is retained in stable_unsplit_run_01.
Baseline deltas are observational here; this run cannot qualify that reference.
No timing or serving speed measurement.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,sys,time,importlib.util,statistics,copy,os,random
from pathlib import Path
from types import MethodType,SimpleNamespace
OWN=Path(__file__).resolve().parent;SWARM=OWN.parents[1];HERE=SWARM/'moe';ROOT=SWARM.parent
sys.path.insert(0,str(HERE/'phase2'))
sys.path.insert(0,str(HERE))
COMBINED=HERE.parent/'coordination/round5_humming_adaptive_fused/candidate_sources'

sys.path.insert(0,str(HERE/'humming_vendor'))
sys.path.insert(0,str(ROOT/'optimization_marlin_tiles_2026-11-15'))

from fixture_tuning_cache import cached_threshold_for_preparation, validate_all_owned_intervals

def main():
    p=argparse.ArgumentParser();p.add_argument('--sglang-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--hidden',type=int,default=4096);p.add_argument('--experts',type=int,default=32);p.add_argument('--tokens',nargs='+',type=int,default=[8,128,512])
    p.add_argument('--cases',nargs='+',default=['4:2','1:4','16:4','32:4']);p.add_argument('--users',nargs='+',type=int,default=[1,4,16,32]);p.add_argument('--keep',nargs='+',type=int,default=[2,4]);p.add_argument('--correctness-only',action='store_true');p.add_argument('--samples',type=int,default=12);p.add_argument('--graph-body',type=int,default=1);p.add_argument('--id-dtype',choices=['int32','int64'],default='int32');p.add_argument('--different-blocks',action='store_true');p.add_argument('--import-only',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
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
    if any(b<=0 for b in args.users) or any(k<=0 or k>=8 for k in args.keep):raise ValueError('Invalid fixed8 prefix fixture')
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
    norm=state['norm_w'];defaults=copy.deepcopy(humm.fp32_configs)
    cases=[tuple(map(int,c.split(':'))) for c in args.cases]
    if any(u<=0 or k<=0 or k>=8 for u,k in cases):raise ValueError('Invalid case')
    report={'status':'running','scope':'DIAGNOSTIC_NOT_PROMOTABLE: fixed original-unsplit reference mask invariance only. Baseline delta measurements are observational; original baseline qualification remains failed. No timing, attention, collectives or serving proof.','source_sha256':{str(path):digest for path,digest in pinned.items()},'rows':[],'journal':[],'baseline_tolerance_checks':[],'changed_tail_strict_checks':[],'reference_contract':'DIAGNOSTIC_NOT_PROMOTABLE. Strict mask-vs-new-unsplit-reference bitwise gate only. Baseline deltas do not qualify reference; original qualification failure remains authoritative.'}
    def save():
      tmp=args.out/'results.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(args.out/'results.json')
    def check(a,b,context):
      if a.numel()==0:r={**context,'elements':0,'bitwise':True,'scope':'No valid raw retained slots'}
      else:
        d=a.float()-b.float();r={**context,'elements':a.numel(),'bitwise':torch.equal(a.view(torch.int16),b.view(torch.int16)),'relative_l2':(d.norm()/b.float().norm().clamp_min(1e-20)).item(),'max_abs':d.abs().max().item(),'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all())}
      report['journal'].append(r);save()
      if not r['bitwise'] or not r.get('finite',True):report['status']='failed_retained_row_bitwise_gate';save();raise AssertionError(r)
      return r
    def capture_one(call):
      for _ in range(2):call()
      torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
      with torch.cuda.graph(g):o=call()
      return g,o
    def sample(g):
      a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
    save()
    for users,case_keep in cases:
      m=8*users
      for keep_n in (case_keep,):
        torch.manual_seed(216+m)
        x=torch.randn((m,args.hidden),device='cuda',dtype=torch.bfloat16)/20
        ids=torch.tensor(stock_make_routes(m,8,args.experts*8,8,'uniform',216+m),device='cuda',dtype=id_dtype)
        local_full=lookup[ids+1];keep=torch.arange(m,device='cuda')%8<keep_n
        local_masked=torch.where(keep[:,None],local_full,-1)
        scores=torch.rand((m,8),device='cuda');scores/=scores.sum(-1,keepdim=True)
        def raw(local):return helper_ns['_humming_expert_outputs_default'](humm,x,local,torch,defer_downscale=True)
        def scored_raw():return helper_ns['_humming_expert_outputs'](humm,x,local_full,torch,defer_downscale=True,global_experts=args.experts*8)
        def finish(value,local):return finalizer_module.fused_post_expert_norm_weighted_sum(value,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=local,expert_output_scale=humm.w2_post_scale)
        def full():return finish(*raw(local_full))
        def scored():return finish(*scored_raw())
        def masked():return finish(*raw(torch.where(keep[:,None],local_full,-1)))
        def baseline_gate(context):
          nr,_=raw(local_full);sr,_=scored_raw();active=(local_full>=0)&(local_full<args.experts)
          pairs=[('raw',nr[active],sr[active]),('final',finish(nr,local_full),finish(sr,local_full))]
          for stage,actual,expected in pairs:
            if not actual.numel():continue
            diff=actual.float()-expected.float();r={**context,'stage':stage,'relative_l2':(diff.norm()/expected.float().norm().clamp_min(1e-20)).item(),'max_abs':diff.abs().max().item(),'bitwise':torch.equal(actual.view(torch.int16),expected.view(torch.int16)),'finite':bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())}
            r['passed']=r['finite'] and r['relative_l2']<=.003 and r['max_abs']<=.03125
            report['baseline_tolerance_checks'].append(r);save()
            r.update({'diagnostic_only':True,'reference_abs_max':expected.float().abs().max().item(),'actual_abs_max':actual.float().abs().max().item(),'reference_l2_norm':expected.float().norm().item(),'different_bf16_elements':int((actual.view(torch.int16)!=expected.view(torch.int16)).sum().item())});save()
        mark('case_begin',users=users,keep=keep_n,m=m)
        context={'users':users,'keep':keep_n,'m':m}
        baseline_gate({**context,'mode':'initial'})
        a,_=raw(local_full);b,_=raw(local_masked);valid=keep[:,None]&(local_full>=0)&(local_full<args.experts)
        raw_check=check(b[valid],a[valid],{**context,'stage':'raw_retained'})
        del a,b,valid
        final_check=check(masked()[keep],full()[keep],{**context,'stage':'final_retained'})
        ga,oa=capture_one(full);gb,ob=capture_one(masked)
        graph_checks=[]
        for seed,pattern in ((223,'uniform'),(227,'hot_expert'),(239,'all_remote')):
          torch.manual_seed(seed);x.copy_(torch.randn_like(x)/20);ids.copy_(torch.tensor(stock_make_routes(m,8,args.experts*8,8,pattern,seed),device='cuda',dtype=id_dtype));local_full.copy_(lookup[ids+1]);local_masked.copy_(torch.where(keep[:,None],local_full,-1))
          baseline_gate({**context,'mode':pattern,'seed':seed})
          ra,_=raw(local_full);rb,_=raw(local_masked);valid=keep[:,None]&(local_full>=0)&(local_full<args.experts)
          check(rb[valid],ra[valid],{**context,'stage':'changed_raw_vs_newref','seed':seed,'pattern':pattern});del ra,rb,valid
          ga.replay();gb.replay();torch.cuda.synchronize()
          graph_checks.append(check(ob[keep],oa[keep],{**context,'stage':'changed_graph_retained','seed':seed,'pattern':pattern}))
        # Preserve prefix values and routes while adversarially replacing unused tails.
        ids.copy_(torch.tensor(stock_make_routes(m,8,args.experts*8,8,'hot_expert',227),device='cuda',dtype=id_dtype));local_full.copy_(lookup[ids+1]);local_masked.copy_(torch.where(keep[:,None],local_full,-1))
        torch.manual_seed(919+m);x.copy_(torch.randn_like(x)/20)
        baseline_gate({**context,'mode':'adversarial_full_hot'})
        old_x=x.clone();old_local=local_full.clone();baseline_kept=full()[keep].clone()
        raw_before,_=raw(local_full);valid=keep[:,None]&(local_full>=0)&(local_full<args.experts);raw_kept=raw_before[valid].clone();del raw_before
        torch.manual_seed(911+m);x[~keep]=torch.randn_like(x[~keep])*7
        local_full[~keep]=-1;local_masked.copy_(torch.where(keep[:,None],local_full,-1))
        baseline_gate({**context,'mode':'adversarial_changed_tail'})
        raw_after,_=raw(local_masked)
        raw_adversary=check(raw_after[valid],raw_kept,{**context,'stage':'arbitrary_tail_raw_vs_newref'})
        graph_before=baseline_kept;gb.replay();torch.cuda.synchronize()
        final_adversary=check(ob[keep],graph_before,{**context,'stage':'arbitrary_tail_graph_vs_newref'})
        report['changed_tail_strict_checks'].append({'users':users,'keep':keep_n,'raw':raw_adversary,'final':final_adversary,'tail_inputs_and_IDs_changed':True})
        x.copy_(old_x);local_full.copy_(old_local);local_masked.copy_(torch.where(keep[:,None],local_full,-1));del old_x,old_local,baseline_kept,raw_kept,raw_after,valid,graph_before
        report['rows'].append({**context,'diagnostic_only':True,'strict_checks_completed':True,'no_timing':True,'raw_check':raw_check,'final_check':final_check,'changed_graph_checks':graph_checks,'arbitrary_tail_raw':raw_adversary,'arbitrary_tail_final':final_adversary})
        assert humm.fp32_configs==defaults
        save();mark('case_complete',users=users,keep=keep_n)
        del ga,gb,oa,ob,x,ids,local_full,local_masked,scores,keep
    report['status']='DIAGNOSTIC_NOT_PROMOTABLE_completed';save();mark('completed')
if __name__=='__main__':main()
