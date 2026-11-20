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

from fused_routes_helper import split_expert_outputs_fused_routes

def main():
    p=argparse.ArgumentParser();p.add_argument('--sglang-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--hidden',type=int,default=4096);p.add_argument('--experts',type=int,default=32);p.add_argument('--tokens',nargs='+',type=int,default=[33,128])
    p.add_argument('--samples',type=int,default=12);p.add_argument('--different-blocks',action='store_true');p.add_argument('--import-only',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
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
    lookup=raw._expert_lookup
    del raw
    norm=state['norm_w']
    if args.samples<6 or args.samples%2:raise ValueError('Even sample count >=6 required')
    defaults=copy.deepcopy(humm.fp32_configs)
    report={'status':'running','argv':sys.argv,'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidate_sha256':{f:hashlib.sha256((OWN/f).read_bytes()).hexdigest() for f in ('fused_routes_helper.py','fused_partition_align.py')},'source_hashes':{str(p):h for p,h in pinned.items()},'gpu':torch.cuda.get_device_name(),'geometry':{'experts':args.experts,'global_experts':args.experts*8,'hidden':args.hidden,'topk':8},'scope':'SM86 source-minimal exact scored helper A versus isolated fused-routing derivative B; shared canonical weights and unchanged BF16 arithmetic; no H200 speed claim','rows':[]}
    def save():(args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    flush=torch.empty(64*1024*1024,device='cuda',dtype=torch.uint8)
    for m in args.tokens:
      plan=choose_partition_configs(humm,m*8,args.experts*8)
      for pattern in ('uniform','hot_rank','hot_expert','mixed','all_remote'):
        x=torch.randn((m,args.hidden),device='cuda',dtype=torch.bfloat16)/20
        ids=torch.tensor(make_routes(m,8,args.experts*8,8,pattern,216),device='cuda',dtype=torch.int32)
        scores=torch.rand((m,8),device='cuda');scores/=scores.sum(-1,keepdim=True)
        def finish(value,local):return finalizer_module.fused_post_expert_norm_weighted_sum(value,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=local,expert_output_scale=humm.w2_post_scale)
        def scored():return finish(*helper_ns['_humming_expert_outputs'](humm,x,lookup[ids+1],torch,defer_downscale=True,global_experts=args.experts*8))
        def candidate():return finish(*split_expert_outputs_fused_routes(helper_ns['_humming_expert_outputs_default'],humm,x,lookup[ids+1],torch,args.experts*8,defer_downscale=True))
        av=scored();bv=candidate();torch.cuda.synchronize();assert torch.equal(av,bv),('eager',m,pattern)
        row={'m':m,'pattern':pattern,'selected_plan':plan,'eager_numeric_equal':True,'replays':[]}
        del av,bv
        ga,oa=capture(torch,scored);gb,ob=capture(torch,candidate)
        for seed,mode in ((221,'uniform'),(227,'hot_rank'),(239,'all_remote'),(251,pattern)):
          torch.manual_seed(seed);x.copy_(torch.randn_like(x)/20);ids.copy_(torch.tensor(make_routes(m,8,args.experts*8,8,mode,seed),device='cuda',dtype=torch.int32));scores.copy_(torch.rand_like(scores));scores.div_(scores.sum(-1,keepdim=True))
          ga.replay();gb.replay();torch.cuda.synchronize();assert torch.equal(oa,ob),('replay',m,mode)
          assert bool(torch.isfinite(oa).all()) and bool(torch.isfinite(ob).all())
          if mode=='all_remote':assert bool((ob==0).all())
          row['replays'].append({'seed':seed,'mode':mode,'numeric_equal':True})
        samples={k:[] for k in ('A1','B1','A2','B2')}
        for order in (list(samples),list(reversed(samples))):
          for label in order:flush.zero_();event_sample(torch,(ga if label[0]=='A' else gb).replay,1)
        for rep in range(args.samples):
          for label in (list(samples) if rep%2==0 else list(reversed(samples))):flush.zero_();samples[label].append(event_sample(torch,(ga if label[0]=='A' else gb).replay,1))
        med={k:statistics.median(v) for k,v in samples.items()};arms={k:statistics.median(samples[k+'1']+samples[k+'2']) for k in 'AB'}
        row.update(samples_us=samples,medians_us=arms,speedup=arms['A']/arms['B'],controls={k:med[k+'1']/med[k+'2'] for k in 'AB'},noise_pass=all(abs(med[k+'1']/med[k+'2']-1)<=.1 for k in 'AB'),passed=True)
        report['rows'].append(row);save();print(json.dumps({k:v for k,v in row.items() if k!='samples_us'}),flush=True)
        del ga,gb,oa,ob;torch.cuda.synchronize();assert humm.fp32_configs==defaults
        if (args.out/'STOP_AFTER_CASE').exists():report['status']='stopped_cooperatively';save();return
    report.update(status='completed',passed=True);save()

if __name__=='__main__':main()
