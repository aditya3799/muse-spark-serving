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
    p.add_argument('--hidden',type=int,default=4096);p.add_argument('--experts',type=int,default=32);p.add_argument('--tokens',nargs='+',type=int,default=[8,128,512])
    p.add_argument('--correctness-only',action='store_true');p.add_argument('--samples',type=int,default=12);p.add_argument('--graph-body',type=int,default=1);p.add_argument('--id-dtype',choices=['int32','int64'],default='int32');p.add_argument('--different-blocks',action='store_true');p.add_argument('--import-only',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
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
    if args.graph_body not in (1,8):raise ValueError('Use body1 latency or body8 throughput')
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

    lookup=raw._expert_lookup
    del raw
    norm=state['norm_w']
    baseline_configs=copy.deepcopy(humm.fp32_configs)
    sm_count=torch.cuda.get_device_properties(0).multi_processor_count
    report={'status':'running','scope':'Same compiled GEMM and arithmetic; only persistent grid size changes. SM86 component evidence, not H200 or serving.',
      'source_sha256':{str(path):digest for path,digest in pinned.items()},
      'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'sm_count':sm_count,'gpu':torch.cuda.get_device_name(),'graph_body':args.graph_body,'rows':[]}
    def save():
      temp=args.out/'results.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(args.out/'results.json')
    def make_view(grid):
      view=copy.copy(humm);view.fp32_configs=copy.deepcopy(humm.fp32_configs)
      for stage,configs in view.fp32_configs.items():
        for lo,hi,c in configs:
          c['num_sms']=grid
          assert c['use_stream_k'] is False and c['use_f16_accum'] is False
      assert view._parameters is humm._parameters
      return view
    views={str(grid):make_view(grid) for grid in sorted(set([max(1,sm_count//4),max(1,sm_count//2),sm_count,sm_count*2]))}
    flush=torch.empty((64*1024*1024,),device='cuda',dtype=torch.uint8)
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    def timed(graph,cold):
      if cold:flush.fill_(17)
      start.record();graph.replay();end.record();end.synchronize()
      return start.elapsed_time(end)*1000/args.graph_body
    def exact(a,b):return torch.equal(a.view(torch.int16),b.view(torch.int16))
    save()
    for m in args.tokens:
      for pattern in ('uniform','hot_expert','all_remote'):
        mark('case_begin',m=m,pattern=pattern)
        torch.manual_seed(216+m)
        x=torch.randn((m,args.hidden),device='cuda',dtype=torch.bfloat16)/20
        ids=torch.tensor(make_routes(m,8,args.experts*8,8,pattern,216+m),device='cuda',dtype=id_dtype)
        local=lookup[ids+1];scores=torch.rand((m,8),device='cuda');scores/=scores.sum(-1,keepdim=True)
        def call(layer):
          value,local_ids=helper_ns['_humming_expert_outputs'](layer,x,local,torch,defer_downscale=True,global_experts=args.experts*8)
          output=finalizer_module.fused_post_expert_norm_weighted_sum(value,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=local_ids,expert_output_scale=layer.w2_post_scale)
          return value,output
        raw_ref,final_ref=call(humm);raw_ref=raw_ref.clone();final_ref=final_ref.clone()
        active=local>=0
        row={'m':m,'pattern':pattern,'active_routes':int(active.sum().item()),'variants':[]}
        report['rows'].append(row);save()
        graph_a,out_a=capture(torch,lambda:call(humm))
        graph_aa,out_aa=capture(torch,lambda:call(humm))
        for label,view in views.items():
          raw_out,final_out=call(view)
          numerical={'raw_active_bitwise':exact(raw_out[active],raw_ref[active]),'final_bitwise':exact(final_out,final_ref)}
          item={'grid_sms':int(label),'numerical':numerical};row['variants'].append(item);save()
          assert all(numerical.values()),item
          graph_b,out_b=capture(torch,lambda:call(view))
          for _ in range(3):graph_a.replay();graph_b.replay();graph_aa.replay()
          torch.cuda.synchronize()
          item['timing']={}
          for cold in (False,True):
            measurements={'A':[],'B':[],'AA':[]}
            graphs={'A':graph_a,'B':graph_b,'AA':graph_aa}
            for sample in range(args.samples):
              order=('A','B','AA') if sample%2==0 else ('AA','B','A')
              for arm in order:measurements[arm].append(timed(graphs[arm],cold))
            medians={arm:statistics.median(v) for arm,v in measurements.items()}
            ratio=medians['A']/medians['AA']
            item['timing']['cold' if cold else 'warm']={'samples_us':measurements,'median_us':medians,
              'speedup':medians['A']/medians['B'],'AA_ratio':ratio,'AA_pass':0.9<=ratio<=1.1}
          # Changing routes and activation values must not depend on captured pointers' old contents.
          for changed in ('all_remote','hot_expert','uniform'):
            torch.manual_seed(913+m);x.copy_(torch.randn_like(x)/20)
            ids.copy_(torch.tensor(make_routes(m,8,args.experts*8,8,changed,913+m),device='cuda',dtype=id_dtype))
            local.copy_(lookup[ids+1]);active=local>=0
            new_raw,new_final=call(humm)
            graph_b.replay();torch.cuda.synchronize()
            assert exact(out_b[0][active],new_raw[active]) and exact(out_b[1],new_final)
          item['changed_input_graph_cases']=3
          # Restore original case before timing the next grid.
          torch.manual_seed(216+m);x.copy_(torch.randn_like(x)/20)
          ids.copy_(torch.tensor(make_routes(m,8,args.experts*8,8,pattern,216+m),device='cuda',dtype=id_dtype));local.copy_(lookup[ids+1]);active=local>=0
          save();del graph_b,out_b
        assert humm.fp32_configs==baseline_configs
        del graph_a,graph_aa,out_a,out_aa,raw_ref,final_ref,raw_out,final_out,new_raw,new_final,local,ids,scores,x
        torch.cuda.synchronize();mark('case_complete',m=m,pattern=pattern)
    report['status']='completed';save()
    for path,digest in pinned.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    mark('completed',rows=len(report['rows']))
if __name__=='__main__':main()

