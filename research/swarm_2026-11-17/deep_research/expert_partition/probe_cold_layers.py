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
    import scalable_partition as candidate
    norm=state['norm_w'];layers=[humm]
    byte_size=lambda layer:sum(t.numel()*t.element_size() for t in list(layer.parameters())+list(layer.buffers()))
    canonical_bytes=byte_size(humm);preflight_rows=[]
    def capacity(stage,required):
      torch.cuda.synchronize();torch.cuda.empty_cache();free,total=torch.cuda.mem_get_info()
      row={'stage':stage,'free_bytes':free,'required_extra_bytes':required,'total_bytes':total,'passed':free>=required};preflight_rows.append(row)
      (args.out/'capacity_preflight.json').write_text(json.dumps(preflight_rows,indent=2)+'\n')
      if not row['passed']:raise RuntimeError(row)
    for index in (1,2):
      capacity('prepare_layer_'+str(index),int(canonical_bytes*2.5)+256*1024**2)
      torch.manual_seed(7100+index);fresh=torch.nn.Module()
      for prefix,n in (('w13',args.hidden*2),('w2',args.hidden)):
        packed=[];scales=[];gs=[]
        for expert in range(args.experts):
          w,s,g,ref=ns[f.name](n,args.hidden,torch.bfloat16);packed.append(w);scales.append(s);gs.append(g);del ref
        for suffix,vs in (('weight',packed),('weight_scale',scales),('weight_scale_2',gs)):
          setattr(fresh,prefix+'_'+suffix,torch.nn.Parameter(torch.stack(vs),requires_grad=False))
        getattr(fresh,prefix+'_weight_scale_2').mul_(torch.linspace(.25,2.0,args.experts,device='cuda'))
        del packed,scales,gs,w,s,g,vs
      # Identical meta/config cache is already populated; no fixture policy change.
      prepared=helper_ns['_prepare_humming_experts'](fresh,torch);del fresh
      assert prepared.fp32_configs==humm.fp32_configs
      layers.append(prepared);mark('distinct_layer_prepared',layer=index,canonical_bytes=byte_size(prepared))
    storages=[]
    for index,layer in enumerate(layers):
      for name,tensor in layer.named_parameters():
        sample=tensor.detach().reshape(-1)[:64].contiguous().view(torch.uint8).cpu().numpy().tobytes()
        storages.append({'layer':index,'name':name,'pointer':tensor.data_ptr(),'bytes':tensor.numel()*tensor.element_size(),'shape':list(tensor.shape),'sample_sha256':hashlib.sha256(sample).hexdigest()})
    assert len({v['pointer'] for v in storages})==len(storages)
    for name in ('w13_weight','w2_weight'):
      assert len({v['sample_sha256'] for v in storages if v['name']==name})==3
    props=torch.cuda.get_device_properties(0)
    l2_bytes=getattr(props,'L2_cache_size',None) or getattr(props,'l2_cache_size',None)
    flush_bytes=64*1024**2
    if l2_bytes is not None and flush_bytes<4*l2_bytes:raise RuntimeError('Flush needs >=4x reported L2')
    capacity('flush_buffer',flush_bytes+128*1024**2)
    flush=torch.zeros((flush_bytes//4,),device='cuda',dtype=torch.int32)
    report={'status':'running','scope':'Three distinct synthetic NVFP4 expert layers at source-default expert geometry; not checkpoint weights, full Muse model, H200, or serving. Same global unique-top8 routes per layer are used by both paths on one localrank.','geometry':{'H':args.hidden,'I':args.hidden,'local_E':args.experts,'global_E':args.experts*8,'layers':3},'layer_storage':storages,'canonical_total_bytes':sum(byte_size(l) for l in layers),'reported_l2_bytes':l2_bytes,'flush_bytes':flush_bytes,'flush_operation':'int32 buffer.add_(1) before start event on same stream; flush excluded from timed interval','cache_scope':'Cache-swept attempt, not a hardware guarantee of zero L2 residency. Distinct layer addresses and working-set bytes are measured separately.','checks':[],'cases':[],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def save():
      tmp=args.out/'results.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(args.out/'results.json')
    def compare(a,b,context):
      if not a.numel():row={**context,'bitwise':True,'finite':True,'scope':'No active slots'}
      else:
        diff=a.float()-b.float();row={**context,'bitwise':torch.equal(a.contiguous().view(torch.int16),b.contiguous().view(torch.int16)),'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all()),'relative_l2':(diff.norm()/b.float().norm().clamp_min(1e-20)).item(),'max_abs':diff.abs().max().item()}
      row['passed']=row['finite'] and row.get('relative_l2',0)<=.003 and row.get('max_abs',0)<=.03125
      report['checks'].append(row);save()
      if not row['passed'] or not row['bitwise']:report.update(status='FAILED_NUMERICAL_GATE',failure=row);save();raise AssertionError(row)
    def finish(value,local,scores,layer):return finalizer_module.fused_post_expert_norm_weighted_sum(value,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=local,expert_output_scale=layer.w2_post_scale)
    def capture(call):
      for _ in range(2):call()
      torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
      with torch.cuda.graph(graph):output=call()
      return graph,output
    def sample(graph,cold):
      if cold:flush.add_(1)
      start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
      start.record();graph.replay();end.record();end.synchronize();return start.elapsed_time(end)*1000
    save()
    for m in args.tokens:
      for pattern in args.patterns:
        # Extra allocation bound includes two separate three-layer graph pools.
        capacity('graphs_'+str(m)+'_'+pattern,2*3*m*8*args.hidden*6+256*1024**2)
        torch.cuda.reset_peak_memory_stats();torch.manual_seed(216+m)
        x=torch.randn((m,args.hidden),device='cuda',dtype=torch.bfloat16)/20
        locals=[];scores=[];global_route_hashes=[]
        for index in range(3):
          cpu=make_routes(m,8,args.experts*8,8,pattern,216+m+index)
          global_route_hashes.append(hashlib.sha256(json.dumps(cpu).encode()).hexdigest())
          ids=torch.tensor(cpu,device='cuda',dtype=id_dtype);locals.append(lookup[ids+1])
          weight=torch.rand((m,8),device='cuda');weight/=weight.sum(-1,keepdim=True);scores.append(weight)
        original_x=x.clone();original_locals=[v.clone() for v in locals];original_scores=[v.clone() for v in scores]
        xa=x.clone();xb=x.clone()
        for index,layer in enumerate(layers):
          local=locals[index]
          av,_=helper_ns['_humming_expert_outputs'](layer,xa,local,torch,defer_downscale=True,global_experts=args.experts*8)
          bv,_=candidate.split_expert_outputs_scalable(helper_ns['_humming_expert_outputs_default'],layer,xb,local,torch,defer_downscale=True,global_experts=args.experts*8)
          active=local>=0;compare(bv[active],av[active],{'m':m,'pattern':pattern,'layer':index,'stage':'raw'})
          ao=finish(av,local,scores[index],layer);bo=finish(bv,local,scores[index],layer)
          compare(bo,ao,{'m':m,'pattern':pattern,'layer':index,'stage':'final'})
          xa=(xa+.01*ao).to(torch.bfloat16);xb=(xb+.01*bo).to(torch.bfloat16)
          compare(xb,xa,{'m':m,'pattern':pattern,'layer':index,'stage':'next_hidden'})
          del av,bv,active,ao,bo
        del xa,xb
        def call_chain(use_candidate):
          current=x;outputs=[]
          for index,layer in enumerate(layers):
            local=locals[index]
            if use_candidate:value,_=candidate.split_expert_outputs_scalable(helper_ns['_humming_expert_outputs_default'],layer,current,local,torch,defer_downscale=True,global_experts=args.experts*8)
            else:value,_=helper_ns['_humming_expert_outputs'](layer,current,local,torch,defer_downscale=True,global_experts=args.experts*8)
            output=finish(value,local,scores[index],layer);outputs.append(output)
            current=(current+.01*output).to(torch.bfloat16)
          return current,tuple(outputs)
        ga,oa=capture(lambda:call_chain(False));gb,ob=capture(lambda:call_chain(True))
        pools=[list(ga.pool()),list(gb.pool())];assert pools[0]!=pools[1]
        for seed,mode in ((551,'uniform'),(557,'hot_rank'),(563,'mixed')):
          torch.manual_seed(seed);x.copy_(torch.randn_like(x)/20)
          for index in range(3):
            cpu=make_routes(m,8,args.experts*8,8,mode,seed+index)
            ids=torch.tensor(cpu,device='cuda',dtype=id_dtype);locals[index].copy_(lookup[ids+1])
          ga.replay();gb.replay();torch.cuda.synchronize()
          compare(ob[0],oa[0],{'m':m,'pattern':pattern,'seed':seed,'stage':'graph_chain_hidden'})
          for index in range(3):compare(ob[1][index],oa[1][index],{'m':m,'pattern':pattern,'seed':seed,'layer':index,'stage':'graph_layer_final'})
        x.copy_(original_x)
        for v,s in zip(locals,original_locals):v.copy_(s)
        for v,s in zip(scores,original_scores):v.copy_(s)
        ga.replay();gb.replay();torch.cuda.synchronize()
        timings=[]
        for cold in (False,True):
          values={name:[] for name in ('A1','B1','A2','B2')}
          for phase in (1,2):
            for sample_id in range(args.samples):
              for name,g in ((('A',ga),('B',gb)) if sample_id%2==0 else (('B',gb),('A',ga))):values[name+str(phase)].append(sample(g,cold))
          med={name:statistics.median(v) for name,v in values.items()};controls={name:med[name+'1']/med[name+'2'] for name in ('A','B')}
          timings.append({'cold':cold,'samples_us':values,'medians_us':med,'controls':controls,'noise_pass':all(.9<=v<=1.1 for v in controls.values()),'speedup':statistics.median(values['A1']+values['A2'])/statistics.median(values['B1']+values['B2'])})
        report['cases'].append({'m':m,'pattern':pattern,'global_route_hashes':global_route_hashes,'distinct_graph_pools':pools,'timings':timings,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()});save();mark('cold_chain_case_complete',m=m,pattern=pattern)
        del ga,gb,g,oa,ob,x,locals,scores,original_x,original_locals,original_scores,ids,weight,local,v,s
    report['status']='completed';save();mark('completed')
if __name__=='__main__':main()


