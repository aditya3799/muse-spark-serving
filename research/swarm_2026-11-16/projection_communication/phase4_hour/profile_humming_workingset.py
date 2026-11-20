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
    p.add_argument('--hidden',type=int,default=4096);p.add_argument('--experts',type=int,default=32);p.add_argument('--tokens',nargs='+',type=int,default=[128])
    p.add_argument('--active-experts',nargs='+',type=int,default=[-1,1,8,32]);p.add_argument('--correctness-only',action='store_true');p.add_argument('--samples',type=int,default=12);p.add_argument('--graph-body',type=int,default=1);p.add_argument('--id-dtype',choices=['int32','int64'],default='int32');p.add_argument('--different-blocks',action='store_true');p.add_argument('--import-only',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    milestones=[]
    def mark(stage,**details):
        item={'stage':stage,'unix_seconds':time.time(),**details}
        torch_module=sys.modules.get('torch')
        if torch_module is not None and torch_module.cuda.is_initialized():
            free,total=torch_module.cuda.mem_get_info()
            item['memory_bytes']={'allocated':torch_module.cuda.memory_allocated(),'reserved':torch_module.cuda.memory_reserved(),'peak_allocated':torch_module.cuda.max_memory_allocated(),'peak_reserved':torch_module.cuda.max_memory_reserved(),'device_free':free,'device_total':total}
        if stage in ('warm_complete','case_complete'):
            import subprocess
            sample=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=timestamp,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
            item['telemetry']={'stdout':sample.stdout,'stderr':sample.stderr,'exit_code':sample.returncode,'scope':'Outside measured timing sample'}
        milestones.append(item)
        (args.out/'milestones.json').write_text(json.dumps(milestones,indent=2)+'\n')
        print(json.dumps(item),flush=True)
    mark('main_entered')
    if any(e < -1 or e > args.experts for e in args.active_experts):raise ValueError('Unsupported distinct-expert fixture')
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

    lookup=raw._expert_lookup
    del raw
    norm=state['norm_w']
    import contextlib,collections
    import sglang.srt.layers.moe.fused_moe_triton as align_module
    import sglang.kernels.ops.activation.activation as activation_module
    torch.cuda.synchronize()
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report={'status':'running','argv':sys.argv,'source_sha256':{str(path):digest for path,digest in pinned.items()},
      'harness_sha256':sha(__file__),'humming_path':str(Path(humming.__file__).resolve()),
      'humming_layer_sha256':sha(Path(humming.__file__).parent/'layer.py'),
      'gpu':torch.cuda.get_device_name(),'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
      'geometry':{'local_experts':args.experts,'global_experts':args.experts*8,'hidden':args.hidden,'intermediate':args.hidden,'topk':8},
      'scope':'Exact scored Humming helper and finalizer only, synthetic controlled routing on one SM86 GPU. Same canonical weights/inputs across eager/graph/instrumented calls. Not whole serving, H200 or eight-rank timings.',
      'timing_definitions':{'host_call_us':'perf_counter around uninstrumented Python call, excluding explicit final synchronization; includes allocation/launch/API work and any implicit stalls',
        'stream_interval_us':'CUDA events around entire call/replay; includes stream idle due to host launch gaps',
        'stage_stream_us':'CUDA events around actual stage functions; instrumentation adds overhead, quantified separately',
        'profiler_device_us':'PyTorch profiler device duration attribution, reported separately; profiler overhead excluded from uninstrumented measures'},
      'rows':[]}
    def save():
      temp=args.out/'results.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(args.out/'results.json')
    save()
    class Recorder:
      def __init__(self):
        self.mode=None;self.records=[];self.cursor=0
        self.pool=[(torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)) for _ in range(32)]
        for a,b in self.pool:a.record();b.record()
        torch.cuda.synchronize()
      def reset(self,mode):self.mode=mode;self.records=[];self.cursor=0
      def call(self,label,fn,*a,**kw):
        if self.mode is None:return fn(*a,**kw)
        if self.mode=='profiler':
          with torch.profiler.record_function('humming_stage/'+label):return fn(*a,**kw)
        assert self.cursor<len(self.pool),'Stage event pool exhausted'
        start,end=self.pool[self.cursor];self.cursor+=1
        start.record();t=time.perf_counter_ns();result=fn(*a,**kw);host=(time.perf_counter_ns()-t)/1000;end.record()
        metadata={k:kw[k] for k in ('sublayer_name','valid_shape_m','top_k','tuning_config') if k in kw}
        self.records.append((label,start,end,host,metadata))
        return result
      def summarize(self,records,host_valid=True):
        sums=collections.defaultdict(lambda:{'calls':0,'stream_us':0,'host_us':0})
        detail=[]
        for label,start,end,host,metadata in records:
          us=start.elapsed_time(end)*1000;sums[label]['calls']+=1;sums[label]['stream_us']+=us;sums[label]['host_us']+=host if host_valid else 0
          detail.append({'name':label,'stream_us':us,'host_us':host if host_valid else None,'metadata':metadata})
        return {'stages':dict(sums),'calls':detail,'total_stage_stream_us':sum(v['stream_us'] for v in sums.values()),'host_valid':host_valid}
    rec=Recorder()
    originals=[]
    def hook(obj,name,label):
      original=getattr(obj,name)
      def wrapped(*a,**kw):return rec.call(label(kw) if callable(label) else label,original,*a,**kw)
      originals.append((obj,name,original,wrapped))
      if obj is HummingMethod:setattr(obj,name,staticmethod(wrapped))
      else:setattr(obj,name,wrapped)
    hook(HummingMethod,'forward_layer',lambda kw:'gemm_'+kw['sublayer_name'])
    hook(align_module,'moe_align_block_size','align')
    hook(split_module,'partition_routes','partition')
    hook(module,'scale_active_rows','up_scale')
    hook(activation_module,'silu_and_mul','activation')
    hook(finalizer_module,'fused_post_expert_norm_weighted_sum','finalize')
    def hooks(enabled):
      for obj,name,original,wrapped in originals:
        if obj is HummingMethod:setattr(obj,name,staticmethod(wrapped) if enabled else classmethod(original.__func__))
        else:setattr(obj,name,wrapped if enabled else original)
    hooks(False)
    defaults=copy.deepcopy(humm.fp32_configs)
    def measure(call,samples):
      values=[];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
      for _ in range(samples):
        torch.cuda.synchronize();a.record();start=time.perf_counter_ns();output=call();host=(time.perf_counter_ns()-start)/1000;b.record();b.synchronize()
        values.append({'host_call_us':host,'stream_interval_us':a.elapsed_time(b)*1000})
      return {'samples':values,'median_host_call_us':statistics.median(v['host_call_us'] for v in values),'median_stream_interval_us':statistics.median(v['stream_interval_us'] for v in values)},output
    def route_fixture(m,distinct,seed):
      if distinct==0:return stock_make_routes(m,8,args.experts*8,8,'uniform',seed)
      rng=random.Random(seed);rows=[]
      for i in range(m):
        if distinct==-1:row=rng.sample(range(args.experts,args.experts*8),8)
        else:
          row=[(i+seed)%distinct]+rng.sample(range(args.experts,args.experts*8),7)
          rng.shuffle(row)
        rows.append(row)
      return rows
    try:
      for m,distinct in [(m,e) for m in args.tokens for e in args.active_experts]:
        pattern='all_remote' if distinct==-1 else 'uniform' if distinct==0 else 'one_local_%d_experts'%distinct
        mark('case_begin',m=m,pattern=pattern)
        torch.manual_seed(216+m)
        x=torch.randn((m,args.hidden),device='cuda',dtype=torch.bfloat16)/20
        ids=torch.tensor(route_fixture(m,distinct,216+m),device='cuda',dtype=id_dtype)
        scores=torch.rand((m,8),device='cuda');scores/=scores.sum(-1,keepdim=True)
        local=lookup[ids+1]
        counts=torch.bincount(local[local>=0].long(),minlength=args.experts).cpu().tolist()
        def call():
          value,local_ids=helper_ns['_humming_expert_outputs'](humm,x,local,torch,defer_downscale=True,global_experts=args.experts*8)
          return finalizer_module.fused_post_expert_norm_weighted_sum(value,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=local_ids,expert_output_scale=humm.w2_post_scale)
        hooks(False);rec.mode=None
        for _ in range(2):call()
        torch.cuda.synchronize();mark('warm_complete',m=m)
        reference=call().clone();torch.cuda.synchronize()
        eager_before,output=measure(call,args.samples)
        assert torch.equal(output.view(torch.int16),reference.view(torch.int16))
        stage_samples=[]
        hooks(True)
        for _ in range(4):
          rec.reset('events');output=call();torch.cuda.synchronize()
          assert torch.equal(output.view(torch.int16),reference.view(torch.int16))
          stage_samples.append(rec.summarize(rec.records))
        rec.mode=None;hooks(False)
        graph,graph_output=capture(torch,call)
        graph_before,_=measure(graph.replay,args.samples)
        assert torch.equal(graph_output.view(torch.int16),reference.view(torch.int16))
        row={'m':m,'pattern':pattern,'requested_distinct_local_experts':distinct,'active_routes':sum(counts),'allocated_route_slots':m*8,'expert_counts':counts,
          'selected_split_plan':choose_partition_configs(humm,m*8,args.experts*8),'eager_before':eager_before,'eager_stages':stage_samples,'plain_graph_before':graph_before,
          'exact_plain_eager_and_graph':True,'graph_body':args.graph_body}
        report['rows'].append(row);save();mark('plain_timings_complete',m=m)
        # Only event records are captured; Python hook overhead is absent on replay.
        hooks(True);rec.reset('events');event_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(event_graph):event_graph_output=call()
        event_records=list(rec.records);rec.mode=None
        event_graph_samples=[]
        for _ in range(4):
          event_graph.replay();torch.cuda.synchronize()
          assert torch.equal(event_graph_output.view(torch.int16),reference.view(torch.int16))
          event_graph_samples.append(rec.summarize(event_records,host_valid=False))
        row['event_graph_stages']=event_graph_samples
        row['event_graph_overhead'],_=measure(event_graph.replay,args.samples)
        row['graph_instrumentation_ratio']=row['event_graph_overhead']['median_stream_interval_us']/graph_before['median_stream_interval_us']
        # One real eager trace and one plain graph trace, separate from all timing samples.
        rec.mode='profiler'
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
          call();torch.cuda.synchronize()
        rec.mode=None
        prof.export_chrome_trace(str(args.out/('eager_m%d_e%d.trace.json'%(m,distinct))))
        def prof_rows(prof):
          return [{'name':e.key,'count':e.count,'cpu_total_us':e.cpu_time_total,'self_cpu_us':e.self_cpu_time_total,
            'device_total_us':getattr(e,'device_time_total',0),'self_device_us':getattr(e,'self_device_time_total',0)} for e in prof.key_averages()]
        row['eager_profiler']=prof_rows(prof)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof_graph:
          graph.replay();torch.cuda.synchronize()
        prof_graph.export_chrome_trace(str(args.out/('graph_m%d_e%d.trace.json'%(m,distinct))))
        row['graph_profiler']=prof_rows(prof_graph)
        # Recheck same-arm drift after instrumented/profiler work.
        hooks(False)
        eager_after,output=measure(call,args.samples);graph_after,_=measure(graph.replay,args.samples)
        row['eager_after']=eager_after;row['plain_graph_after']=graph_after
        row['same_arm_controls']={kind:before['median_stream_interval_us']/after['median_stream_interval_us'] for kind,before,after in [('eager',eager_before,eager_after),('graph',graph_before,graph_after)]}
        row['same_arm_noise_pass']=all(0.9<=r<=1.1 for r in row['same_arm_controls'].values())
        row['graph_stage_evidence_usable']=row['same_arm_noise_pass'] and row['graph_instrumentation_ratio']<=1.1
        # Changed input/route graph comparison prevents stale-output evidence.
        torch.manual_seed(427+m);x.copy_(torch.randn_like(x)/20)
        ids.copy_(torch.tensor(route_fixture(m,distinct,427+m),device='cuda',dtype=id_dtype));local.copy_(lookup[ids+1])
        new_ref=call().clone();graph.replay();event_graph.replay();torch.cuda.synchronize()
        row['changed_input_graph_exact']=torch.equal(new_ref.view(torch.int16),graph_output.view(torch.int16)) and torch.equal(new_ref.view(torch.int16),event_graph_output.view(torch.int16))
        assert row['changed_input_graph_exact']
        row['memory']={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated()}
        assert humm.fp32_configs==defaults
        save();mark('case_complete',m=m,pattern=pattern,noise_pass=row['same_arm_noise_pass'],graph_event_overhead=row['graph_instrumentation_ratio'])
        del graph,event_graph,graph_output,event_graph_output,reference,new_ref,output,local,ids,scores,x
        torch.cuda.synchronize()
      report['status']='completed';save()
    finally:
      rec.mode=None
      for obj,name,original,wrapped in reversed(originals):
        if obj is HummingMethod:setattr(obj,name,classmethod(original.__func__))
        else:setattr(obj,name,original)
    for path,digest in pinned.items():assert sha(path)==digest
    mark('completed',rows=len(report['rows']))
if __name__=='__main__':main()
