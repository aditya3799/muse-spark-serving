"""Pinned Humming 0.1.12 adapter experiment with stock BF16 scale boundaries."""
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
    p.add_argument('--users',nargs='+',type=int,default=[1,4,16,32]);p.add_argument('--keep',nargs='+',type=int,default=[2,4]);p.add_argument('--correctness-only',action='store_true');p.add_argument('--samples',type=int,default=12);p.add_argument('--graph-body',type=int,default=1);p.add_argument('--id-dtype',choices=['int32','int64'],default='int32');p.add_argument('--different-blocks',action='store_true');p.add_argument('--import-only',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
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


    spec=importlib.util.spec_from_file_location('tailmask_preserve_partition',OWN/'tailmask_preserve_partition.py')
    preserve=importlib.util.module_from_spec(spec);sys.modules[spec.name]=preserve;spec.loader.exec_module(preserve)
    lookup=raw._expert_lookup;del raw
    norm=state['norm_w'];defaults=copy.deepcopy(humm.fp32_configs)
    report={'status':'running','scope':'Exact scored Humming helper only; full8rows perrequest remains in both arms. Tail local IDs become-1, retained row strides, inputs, weights and logicalM unchanged. No attention, collectives or full serving proof.','source_sha256':{str(path):digest for path,digest in pinned.items()},'rows':[],'journal':[],'adversarial_tail_changes':[],'candidate_sha256':hashlib.sha256((OWN/'tailmask_preserve_partition.py').read_bytes()).hexdigest(),'multilayer_status':'NOT_SOLVED: candidate later-layer tail histogram can differ from original baseline'}
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
    for users in args.users:
      m=8*users
      for keep_n in args.keep:
        torch.manual_seed(216+m)
        x=torch.randn((m,args.hidden),device='cuda',dtype=torch.bfloat16)/20
        ids=torch.tensor(stock_make_routes(m,8,args.experts*8,8,'uniform',216+m),device='cuda',dtype=id_dtype)
        local_full=lookup[ids+1];keep=torch.arange(m,device='cuda')%8<keep_n
        local_masked=torch.where(keep[:,None],local_full,-1)
        scores=torch.rand((m,8),device='cuda');scores/=scores.sum(-1,keepdim=True)
        def raw(local):return helper_ns['_humming_expert_outputs'](humm,x,local,torch,defer_downscale=True,global_experts=args.experts*8)
        def finish(value,local):return finalizer_module.fused_post_expert_norm_weighted_sum(value,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=local,expert_output_scale=humm.w2_post_scale)
        def full():return finish(*raw(local_full))
        def masked_raw(classification_ids=None):return preserve.masked_expert_outputs(helper_ns['_humming_expert_outputs_default'],humm,x,local_full if classification_ids is None else classification_ids,keep,torch,global_experts=args.experts*8,defer_downscale=True)
        def masked():return finish(*masked_raw())
        mark('case_begin',users=users,keep=keep_n,m=m)
        context={'users':users,'keep':keep_n,'m':m}
        a,_=raw(local_full);b,_=masked_raw();valid=keep[:,None]&(local_full>=0)&(local_full<args.experts)
        raw_check=check(b[valid],a[valid],{**context,'stage':'raw_retained'})
        del a,b,valid
        final_check=check(masked()[keep],full()[keep],{**context,'stage':'final_retained'})
        ga,oa=capture_one(full);gb,ob=capture_one(masked)
        graph_checks=[]
        for seed,pattern in ((223,'uniform'),(227,'hot_expert'),(239,'all_remote')):
          torch.manual_seed(seed);x.copy_(torch.randn_like(x)/20);ids.copy_(torch.tensor(stock_make_routes(m,8,args.experts*8,8,pattern,seed),device='cuda',dtype=id_dtype));local_full.copy_(lookup[ids+1]);local_masked.copy_(torch.where(keep[:,None],local_full,-1))
          ra,_=raw(local_full);rb,_=masked_raw();valid=keep[:,None]&(local_full>=0)&(local_full<args.experts)
          check(rb[valid],ra[valid],{**context,'stage':'changed_raw_retained','seed':seed,'pattern':pattern});del ra,rb,valid
          ga.replay();gb.replay();torch.cuda.synchronize()
          graph_checks.append(check(ob[keep],oa[keep],{**context,'stage':'changed_graph_retained','seed':seed,'pattern':pattern}))
        ids.copy_(torch.tensor(stock_make_routes(m,8,args.experts*8,8,'uniform',216+m),device='cuda',dtype=id_dtype));local_full.copy_(lookup[ids+1]);local_masked.copy_(torch.where(keep[:,None],local_full,-1));ga.replay();gb.replay();torch.cuda.synchronize()
        samples={name:[] for name in ('A1','B1','A2','B2')}
        for phase in (1,2):
          for i in range(args.samples):
            for name,g in ((('A',ga),('B',gb)) if i%2==0 else (('B',gb),('A',ga))):samples[name+str(phase)].append(sample(g))
        med={name:statistics.median(v) for name,v in samples.items()};controls={'A':med['A1']/med['A2'],'B':med['B1']/med['B2']}
        counts={name:torch.bincount(local[local>=0].long(),minlength=args.experts).cpu().tolist() for name,local in (('full',local_full),('masked',local_masked))}
        report['rows'].append({**context,'selected_plan':choose_partition_configs(humm,m*8,args.experts*8),'route_histograms':counts,'global_route_histograms':{'full':torch.bincount(ids.flatten().long(),minlength=args.experts*8).cpu().tolist(),'masked':torch.bincount(ids[keep].flatten().long(),minlength=args.experts*8).cpu().tolist()},'mask_construction_in_timed_graph':True,'raw_check':raw_check,'final_check':final_check,'changed_graph_checks':graph_checks,'samples_us':samples,'medians_us':med,'controls':controls,'noise_pass':all(.9<=v<=1.1 for v in controls.values()),'speedup':statistics.median(samples['A1']+samples['A2'])/statistics.median(samples['B1']+samples['B2']),'tail_output_ignored':True})
        # Adversary: retained inputs/IDs identical, but unused-tail IDs differ.
        # This models the full-histogram mismatch after an earlier masked layer.
        selected=choose_partition_configs(humm,m*8,args.experts*8)
        if selected is not None:
          cutoff=selected['cutoff'];tail_rows=[i for i in range(m) if i%8>=keep_n]
          if len(tail_rows)>=cutoff:
            adv_full=torch.full_like(local_full,-1);adv_full[0,0]=0
            adv_full[torch.tensor(tail_rows[:cutoff],device='cuda'),0]=0
            adv_changed=torch.where(keep[:,None],adv_full,-1)
            original=finish(*raw(adv_full));candidate=finish(*masked_raw(adv_changed))
            diff=candidate[keep].float()-original[keep].float()
            adversary={**context,'cutoff':cutoff,'baseline_expert0_count':cutoff+1,'candidate_expert0_count':1,'retained_inputs_ids_identical':True,'bitwise':torch.equal(candidate[keep].view(torch.int16),original[keep].view(torch.int16)),'relative_l2':(diff.norm()/original[keep].float().norm().clamp_min(1e-20)).item(),'max_abs':diff.abs().max().item(),'interpretation':'Any mismatch blocks multilayer use; same-input one-layer equality is insufficient.'}
            report['adversarial_tail_changes'].append(adversary);save()
            del adv_full,adv_changed,original,candidate,diff
        assert humm.fp32_configs==defaults
        save();mark('case_complete',users=users,keep=keep_n)
        del ga,gb,oa,ob,x,ids,local_full,local_masked,scores,keep
    report['status']='completed';save();mark('completed')
if __name__=='__main__':main()
