"""Pinned Humming 0.1.12 adapter experiment with stock BF16 scale boundaries."""
from __future__ import annotations
import argparse,ast,hashlib,json,sys,time,importlib.util,statistics
from pathlib import Path
from types import MethodType,SimpleNamespace
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/'humming_vendor'))
sys.path.insert(0,str(ROOT/'optimization_marlin_tiles_2026-11-15'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--sglang-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--hidden',type=int,default=512);p.add_argument('--experts',type=int,default=4);p.add_argument('--tokens',nargs='+',type=int,default=[1,8,33,128])
    p.add_argument('--samples',type=int,default=11);p.add_argument('--different-blocks',action='store_true');p.add_argument('--import-only',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
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
    from tile_study import make_routes
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
    export=HERE/'serving_humming_v1/python/sglang/srt/layers/muse_spark/muse_spark_nvfp4_moe.py'
    frozen_tree=ast.parse(export.read_text(encoding='utf-8'))
    helper_nodes=[n for n in frozen_tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_prepare_humming_experts','_humming_expert_outputs')]
    helper_ns={'torch':torch}
    exec(compile(ast.Module(body=helper_nodes,type_ignores=[]),str(export),'exec'),helper_ns)

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
    report={'status':'running','argv':sys.argv,'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'geometry':{'local_experts':args.experts,'global_experts':args.experts*8,'hidden':args.hidden,'intermediate':args.hidden,'top_k':8,'tokens':args.tokens},'runtime':runtime,'humming_version':'0.1.12','gpu':torch.cuda.get_device_name(),'rows':[],'provenance':provenance,'hot_rank_is_fallback':args.experts<8,
            'component_gate':{'relative_l2_max':.003,'absolute_max':.03125},
            'scope':'exact frozen serving helpers; scaled globals, changed graph inputs, FP32 accumulation and no Stream-K', 'artifact_sha256':hashlib.sha256(export.read_bytes()).hexdigest(), 'different_blocks':args.different_blocks}
    flush=torch.empty(64*1024*1024,device='cuda',dtype=torch.uint8)
    report['timing_scope']='SM86 local single-shard CUDA graph GPU latency only; excludes host dispatch, model communication and full serving; not H200 performance proof'
    report['timing_protocol']={'flush_bytes':flush.numel(),'one_replay_per_sample':True,'alternating_order':['A1,B1,A2,B2','B2,A2,B1,A1'],'A_labels':'same stock graph twice','B_labels':'same exact frozen Humming graph twice','samples_per_label':args.samples,'reference':'stock Marlin, not compact32 fallback'}
    if args.samples<5:raise ValueError('At least five timing samples required')
    original_w2_configs=humm.fp32_configs['w2']
    for m in args.tokens:
      humm.fp32_configs['w2']=original_w2_configs
      for pattern in ('uniform','hot_rank','hot_expert','all_remote'):
        x=torch.randn((m,args.hidden),device='cuda',dtype=torch.bfloat16)/20
        ids=torch.tensor(make_routes(m,8,args.experts*8,8,pattern,216),device='cuda',dtype=torch.int32)
        scores=torch.rand((m,8),device='cuda');scores/=scores.sum(-1,keepdim=True)
        def finish(v,li):return fused_post_expert_norm_weighted_sum(v,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=li)
        def baseline():return finish(*method(x,scores,ids))
        if args.different_blocks:
            first=next(c for lo,hi,c in humm.fp32_configs['w13'] if lo<ids.numel()<=hi)
            different=next(c for lo,hi,c in original_w2_configs if c['block_shape'][0]!=first['block_shape'][0])
            humm.fp32_configs['w2']=[(0,1<<30,different)]
        def candidate():return finish(*helper_ns['_humming_expert_outputs'](humm,x,raw._expert_lookup[ids+1],torch))
        print(f'case m={m} pattern={pattern}',flush=True)
        b=baseline();a=candidate();torch.cuda.synchronize()
        eager_diff=a.float()-b.float()
        eager_rel=(eager_diff.norm()/b.float().norm().clamp_min(1e-20)).item()
        eager_abs=eager_diff.abs().max().item()
        assert torch.isfinite(a).all() and eager_rel<=.003 and eager_abs<=.03125,(m,pattern,'eager',eager_rel,eager_abs)
        del eager_diff
        replay_rows=[]
        # Every graph also transitions to all-remote routes, then back to its pattern.
        gb,ob=capture(torch,baseline);ga,oa=capture(torch,candidate)
        for replay_seed in (221,239,251):
            torch.manual_seed(replay_seed)
            x.copy_(torch.randn_like(x)/20)
            replay_pattern='all_remote' if replay_seed==239 else pattern
            ids.copy_(torch.tensor(make_routes(m,8,args.experts*8,8,replay_pattern,replay_seed),device='cuda',dtype=torch.int32))
            scores.copy_(torch.rand_like(scores));scores.div_(scores.sum(-1,keepdim=True))
            gb.replay();ga.replay();torch.cuda.synchronize()
            dd=oa.float()-ob.float()
            rr=(dd.norm()/ob.float().norm().clamp_min(1e-20)).item();aa=dd.abs().max().item()
            valid=bool(torch.isfinite(oa).all()) and rr<=.003 and aa<=.03125
            if replay_pattern=='all_remote':valid=valid and bool((oa==0).all()) and bool((ob==0).all())
            replay_rows.append({'seed':replay_seed,'routing':replay_pattern,'relative_l2':rr,'max_abs':aa,'passed':valid})
            if not valid:
                report['failure']={'m':m,'pattern':pattern,'replays':replay_rows}
                (args.out/'results.json').write_text(json.dumps(report,indent=2))
                raise AssertionError((m,pattern,replay_seed,rr,aa))
        b=ob;a=oa
        diff=a.float()-b.float();rel=(diff.norm()/b.float().norm().clamp_min(1e-20)).item();ab=diff.abs().max().item()
        row={'m':m,'pattern':pattern,'eager_relative_l2':eager_rel,'eager_max_abs':eager_abs,'replays':replay_rows,'relative_l2':rel,'max_abs':ab,'passed':bool(torch.isfinite(a).all()) and rel<=.003 and ab<=.03125}
        samples={label:[] for label in ('A1','B1','A2','B2')}
        for rep in range(args.samples):
            order=list(samples) if rep%2==0 else list(reversed(samples))
            for label in order:
                flush.zero_()
                samples[label].append(event_sample(torch,gb.replay if label.startswith('A') else ga.replay,1))
        medians={label:statistics.median(values) for label,values in samples.items()}
        row['cold_samples_us']=samples
        row['cold_median_us']=medians
        row['cold_stock_median_us']=statistics.median(samples['A1']+samples['A2'])
        row['cold_humming_median_us']=statistics.median(samples['B1']+samples['B2'])
        row['cold_speedup']=row['cold_stock_median_us']/row['cold_humming_median_us']
        row['aa_ratio']=medians['A1']/medians['A2']
        row['bb_ratio']=medians['B1']/medians['B2']
        row['noise_within_10pct']=abs(row['aa_ratio']-1)<=.1 and abs(row['bb_ratio']-1)<=.1
        report['rows'].append(row);(args.out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
        del gb,ga,ob,oa,b,a,dd,diff
        torch.cuda.synchronize()
        if (args.out/'STOP_AFTER_CASE').exists():
            report['stopped_cooperatively']=True
            report['status']='stopped_cooperatively'
            (args.out/'results.json').write_text(json.dumps(report,indent=2))
            return
    report['passed']=all(r['passed'] for r in report['rows']);report['status']='completed';(args.out/'results.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()



