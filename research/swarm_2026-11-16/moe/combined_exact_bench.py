"""Pinned Humming 0.1.12 adapter experiment with stock BF16 scale boundaries."""
from __future__ import annotations
import argparse,ast,hashlib,json,sys,time,importlib.util,statistics,copy,os,random
from pathlib import Path
from types import MethodType,SimpleNamespace
OWN=Path(__file__).resolve().parent;HERE=OWN.parent/'moe';ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
COMBINED=HERE.parent/'projection_communication/adaptive_split_v3/serving_export'

sys.path.insert(0,str(HERE/'humming_vendor'))
sys.path.insert(0,str(ROOT/'optimization_marlin_tiles_2026-11-15'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--sglang-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--hidden',type=int,default=4096);p.add_argument('--experts',type=int,default=32);p.add_argument('--tokens',nargs='+',type=int,default=[33,128])
    p.add_argument('--samples',type=int,default=12);p.add_argument('--different-blocks',action='store_true');p.add_argument('--import-only',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if os.environ.get('MAX_JOBS')!='1' or os.environ.get('HUMMING_DISABLE_PARALLEL_BUILD')!='1':raise RuntimeError('Serial compiler environment required')
    if args.different_blocks:raise ValueError('EP experiment cannot alter independent block policy')
    manifest=json.loads((COMBINED.parent/'manifest.json').read_text())
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
    from marlin_bench import load_method,capture,event_sample
    from tile_study import make_routes as stock_make_routes
    def make_routes(m,topk,experts,ep,pattern,seed):
        if pattern!='mixed':return stock_make_routes(m,topk,experts,ep,pattern,seed)
        rng=random.Random(seed)
        return [rng.sample(range(topk),topk) if i<m//2 else rng.sample(range(experts),topk) for i in range(m)]
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
    from sglang.kernels.ops.moe.muse_spark_post_expert_finalize_triton import fused_post_expert_norm_weighted_sum
    tree=ast.parse((args.sglang_root/'python/sglang/test/test_marlin_utils.py').read_text())
    f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_nvfp4_weight_and_ref')
    ns={'torch':torch};exec(compile(ast.Module(body=[f],type_ignores=[]),'<stock-generator>','exec'),ns)
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
    print('Preparing Humming canonical weights',flush=True)
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


    norm=torch.linspace(-.5,.5,args.hidden,device='cuda',dtype=torch.bfloat16)
    weights={'post_norm_w':norm}
    for prefix in ('w13','w2'):
        weights[prefix]=getattr(raw,prefix+'_weight')
        weights[prefix+'_sf']=getattr(raw,prefix+'_weight_scale')
        gs=getattr(raw,prefix+'_weight_scale_2')
        gs.mul_(torch.linspace(.25,2.0,args.experts,device='cuda'))
        weights[prefix+'_gs']=gs
    state={'layer':helper_ns['_prepare_humming_experts'](raw,torch),'norm_w':(norm.float()+1).to(torch.bfloat16)}
    del weights
    humm=state['layer']
    print('Preparing Marlin reference weights',flush=True)
    prepare_moe_nvfp4_layer_for_marlin(raw)
    method=MethodType(load_method(args.sglang_root,'baseline',torch)[0],raw)
    norm=state['norm_w']
    if args.samples<6 or args.samples%2:raise ValueError('Use an even sample count >=6 for balanced order')
    if args.experts<8:raise ValueError('True hot-rank requires local experts >= topk8')
    defaults=copy.deepcopy(humm.fp32_configs)
    report={'status':'running','argv':sys.argv,'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'combined_manifest_sha256':hashlib.sha256((COMBINED.parent/'manifest.json').read_bytes()).hexdigest(),'source_hashes':{str(p):d for p,d in pinned.items()},'runtime':runtime,'provenance':provenance,'gpu':torch.cuda.get_device_name(),'geometry':{'local_experts':args.experts,'global_experts':args.experts*8,'hidden':args.hidden,'intermediate':args.hidden,'top_k':8,'tokens':args.tokens},'scope':'SM86 local single-shard experiment: A stock Marlin, B exact v3 default helper with fused finalizer, C actual combined adaptive helper with fused finalizer. Additional nondeferred and absent-metadata paths checked for correctness. No H200/serving proof.','gate':{'relative_l2':.003,'max_abs':.03125},'rows':[],'timing_protocol':{'labels':['A1','B1','C1','A2','B2','C2'],'reverse_every_other_rep':True,'samples_per_label':args.samples,'flush_bytes':64*1024*1024,'one_replay_per_event':True,'unmeasured_full_orders':2,'stock_is_not_compact32':True}}
    flush=torch.empty(64*1024*1024,device='cuda',dtype=torch.uint8)
    def save(): (args.out/'results.json').write_text(json.dumps(report,indent=2))
    def screen(value,reference,label,remote=False):
        delta=value.float()-reference.float();rel=(delta.norm()/reference.float().norm().clamp_min(1e-20)).item();ab=delta.abs().max().item()
        passed=bool(torch.isfinite(value).all()) and bool(torch.isfinite(reference).all()) and rel<=.003 and ab<=.03125
        if remote:passed=passed and bool((value==0).all()) and bool((reference==0).all())
        result={'label':label,'relative_l2':rel,'max_abs':ab,'passed':passed}
        if not passed:report['status']='numerical_failure';report['failure']=result;save();raise AssertionError(result)
        return result
    for m in args.tokens:
      selections=choose_partition_configs(humm,m*8,args.experts*8)
      actual_default={s:next(copy.deepcopy(c) for lo,hi,c in cfg if lo<m*8<=hi) for s,cfg in defaults.items()}
      for pattern in ('uniform','hot_rank','hot_expert','mixed','all_remote'):
        x=torch.randn((m,args.hidden),device='cuda',dtype=torch.bfloat16)/20
        ids=torch.tensor(make_routes(m,8,args.experts*8,8,pattern,216),device='cuda',dtype=torch.int32)
        scores=torch.rand((m,8),device='cuda');scores/=scores.sum(-1,keepdim=True)
        def finish(v,li):return fused_post_expert_norm_weighted_sum(v,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=li)
        def stock():return finish(*method(x,scores,ids))
        def finish_fused(value,local):return finalizer_module.fused_post_expert_norm_weighted_sum(value,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=local,expert_output_scale=humm.w2_post_scale)
        def default_v2():return finish_fused(*helper_ns['_humming_expert_outputs_default'](humm,x,raw._expert_lookup[ids+1],torch,defer_downscale=True))
        def split_v2():return finish_fused(*helper_ns['_humming_expert_outputs'](humm,x,raw._expert_lookup[ids+1],torch,defer_downscale=True,global_experts=args.experts*8))
        def combined_nondeferred():return finish(*helper_ns['_humming_expert_outputs'](humm,x,raw._expert_lookup[ids+1],torch,global_experts=args.experts*8))
        row={'m':m,'pattern':pattern,'actual_routes':m*8,'selected_default':actual_default,'selected_split':selections,'eager':[],'replays':[]}
        print(json.dumps({'m':m,'pattern':pattern,'default':actual_default,'split':selections}),flush=True)
        if selections is not None:
            local=raw._expert_lookup[ids+1]
            li,hi,counts=partition_routes(local,args.experts,selections['cutoff'])
            valid_ids=local[local>=0].long()
            expected_counts=torch.bincount(valid_ids,minlength=args.experts).to(torch.int32)
            assert torch.equal(counts,expected_counts)
            local_safe=local.clamp_min(0).long()
            light_mask=(local>=0)&(expected_counts[local_safe]<=selections['cutoff'])
            heavy_mask=(local>=0)&~light_mask
            assert torch.equal(li,torch.where(light_mask,local,-1))
            assert torch.equal(hi,torch.where(heavy_mask,local,-1))
            row['partition_counts']={'light_rows':int(light_mask.sum()),'heavy_rows':int(heavy_mask.sum()),'expert_counts':counts.cpu().tolist()}
            del local,li,hi,counts,valid_ids,expected_counts,local_safe,light_mask,heavy_mask
        a=stock();b=default_v2();c=split_v2();torch.cuda.synchronize()
        for val,ref,label in ((b,a,'B_vs_A'),(c,a,'C_vs_A'),(c,b,'C_vs_B')):row['eager'].append(screen(val,ref,label,pattern=='all_remote'))
        separate=combined_nondeferred();torch.cuda.synchronize()
        assert torch.equal(c,separate),('combined defer differs',m,pattern)
        row['combined_defer_exact_equal']=True
        row['absent_metadata_checks']=[]
        for deferred in (False,True):
            missing=helper_ns['_humming_expert_outputs'](humm,x,raw._expert_lookup[ids+1],torch,defer_downscale=deferred)
            missing_output=finish_fused(*missing) if deferred else finish(*missing)
            assert torch.equal(missing_output,b),('absent metadata differs',m,pattern,deferred)
            row['absent_metadata_checks'].append({'defer_downscale':deferred,'exact_equal_default':True})
        del a,b,c,val,ref,separate,missing,missing_output
        ga,oa=capture(torch,stock);gb,ob=capture(torch,default_v2);gc,oc=capture(torch,split_v2);gd,od=capture(torch,combined_nondeferred)
        assert humm.fp32_configs==defaults,'Base config table mutated during capture'
        for seed,replay_pattern in ((221,'uniform'),(227,'hot_rank'),(239,'all_remote'),(251,pattern)):
            torch.manual_seed(seed);x.copy_(torch.randn_like(x)/20)
            ids.copy_(torch.tensor(make_routes(m,8,args.experts*8,8,replay_pattern,seed),device='cuda',dtype=torch.int32))
            scores.copy_(torch.rand_like(scores));scores.div_(scores.sum(-1,keepdim=True))
            ga.replay();gb.replay();gc.replay();gd.replay();torch.cuda.synchronize()
            assert torch.equal(oc,od),('combined graph defer differs',m,pattern,seed)
            checks=[screen(v,r,l,replay_pattern=='all_remote') for v,r,l in ((ob,oa,'B_vs_A'),(oc,oa,'C_vs_A'),(oc,ob,'C_vs_B'))]
            row['replays'].append({'seed':seed,'pattern':replay_pattern,'checks':checks,'combined_defer_exact_equal':True})
        samples={label:[] for label in ('A1','B1','C1','A2','B2','C2')}
        # Two unmeasured full forward/reverse orders initialize event/launch paths equally.
        for order in (list(samples),list(reversed(samples))):
            for label in order:
                flush.zero_();event_sample(torch,{'A':ga,'B':gb,'C':gc}[label[0]].replay,1)
        for rep in range(args.samples):
            for label in (list(samples) if rep%2==0 else list(reversed(samples))):
                flush.zero_();samples[label].append(event_sample(torch,{'A':ga,'B':gb,'C':gc}[label[0]].replay,1))
        med={label:statistics.median(v) for label,v in samples.items()};arms={k:statistics.median(samples[k+'1']+samples[k+'2']) for k in 'ABC'}
        row.update(samples_us=samples,label_medians_us=med,arm_medians_us=arms,combined_vs_stock=arms['A']/arms['C'],combined_vs_default=arms['B']/arms['C'],controls={k:med[k+'1']/med[k+'2'] for k in 'ABC'},noise_pass=all(abs(med[k+'1']/med[k+'2']-1)<=.1 for k in 'ABC'),passed=True)
        report['rows'].append(row);save();print(json.dumps({k:v for k,v in row.items() if k!='samples_us'}),flush=True)
        del ga,gb,gc,gd,oa,ob,oc,od;torch.cuda.synchronize()
        assert humm.fp32_configs==defaults,'Base config table mutated'
        if (args.out/'STOP_AFTER_CASE').exists():report['status']='stopped_cooperatively';save();return
    report['status']='completed';report['passed']=True;save()

if __name__=='__main__':main()
