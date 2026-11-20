"""Pinned Humming 0.1.12 adapter experiment with stock BF16 scale boundaries."""
from __future__ import annotations
import argparse,ast,hashlib,json,sys,time
from pathlib import Path
from types import MethodType,SimpleNamespace
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/'humming_vendor'))
sys.path.insert(0,str(ROOT/'optimization_marlin_tiles_2026-11-15'))

def prepare_humming(raw,torch):
    from humming import dtypes
    from humming.config import WeightScale2Type,WeightScaleType
    from humming.layer import HummingMethod
    from humming.schema import HummingInputSchema,HummingWeightSchema
    layer=torch.nn.Module()
    for prefix in ('w13','w2'):
        w=getattr(raw,prefix+'_weight');sf=getattr(raw,prefix+'_weight_scale')
        gs=getattr(raw,prefix+'_weight_scale_2')
        e,n,k2=w.shape
        layer.register_parameter(prefix+'_weight',torch.nn.Parameter(w.view(torch.int32),requires_grad=False))
        layer.register_parameter(prefix+'_weight_scale',torch.nn.Parameter(sf,requires_grad=False))
        # Marlin first rounds the accumulator to BF16, then multiplies by a BF16
        # global scale. Native Humming's FP32 epilogue scale has another rounding
        # boundary, so apply this factor separately after each BF16 GEMM output.
        layer.register_buffer(prefix+'_post_scale',gs.to(torch.bfloat16).contiguous())
        schema=HummingWeightSchema(b_dtype=dtypes.float4e2m1,bs_dtype=dtypes.float8e4m3,
                weight_scale_group_size=16,weight_scale_type=WeightScaleType.GROUP,
                weight_scale_2_type=WeightScale2Type.NONE)
        meta=HummingMethod.prepare_layer_meta(layer,shape_n=n,shape_k=k2*2,weight_schema=schema,
                input_schema=HummingInputSchema(a_dtype=dtypes.bfloat16),num_experts=e,
                torch_dtype=torch.bfloat16,sublayer_name=prefix)
        assert meta.a_dtype==dtypes.bfloat16 and meta.c_dtype==dtypes.bfloat16
        assert not meta.use_int_weight_scale and not meta.has_input_scale
        HummingMethod.transform_humming_layer(layer,sublayer_name=prefix)
    layer.register_buffer('locks',torch.zeros(max(2048,torch.cuda.get_device_properties(0).multi_processor_count*8),device='cuda',dtype=torch.int32))
    import copy
    layer.fp32_configs={}
    for stage in ('w13','w2'):
        default=HummingMethod.get_default_tuning_configs(layer,use_f16_accum=False,gemm_type='indexed',sublayer_name=stage)
        snapshot=copy.deepcopy(default)
        owned=copy.deepcopy(default)
        for lo,hi,c in owned:
            c['use_stream_k']=False
            c['use_f16_accum']=False
        assert default==snapshot, 'Default tuning cache must stay unchanged'
        layer.fp32_configs[stage]=owned
    return layer


def indexed(layer,x,local_ids,torch,retain=False):
    from humming.layer import HummingMethod
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.kernels.ops.activation.activation import silu_and_mul
    configs=layer.fp32_configs
    valid=local_ids.numel()
    selected={s:next(c for lo,hi,c in cfg if lo<valid<=hi) for s,cfg in configs.items()}
    assert all(c['use_stream_k'] is False and c['use_f16_accum'] is False for c in selected.values())
    block=selected['w13']['block_shape'][0]
    sorted_ids,expert_ids,padded=moe_align_block_size(local_ids,block,layer.w13_post_scale.numel(),ignore_invalid_expert=True)
    m,k=x.shape;topk=local_ids.shape[1];i=layer.humming_metas['w2'].shape_k
    cache13=torch.empty((m*topk*max(2*i,k),),device=x.device,dtype=x.dtype)
    cache1=cache13[:m*topk*2*i].view(m*topk,2*i)
    common=dict(sorted_ids=sorted_ids,expert_ids=expert_ids,num_tokens_padded=padded,
                valid_shape_m=valid,compute_config={'gemm_type':'indexed','use_f16_accum':False})
    HummingMethod.forward_layer(layer,inputs=x,outputs=cache1,top_k=topk,sublayer_name='w13',tuning_config=configs['w13'],**common)
    safe=local_ids.flatten().clamp_min(0).long()
    cache1.mul_(layer.w13_post_scale[safe,None])
    cache2=torch.empty((m*topk,i),device=x.device,dtype=x.dtype)
    silu_and_mul(cache1,cache2,expert_ids=local_ids.flatten(),expert_step=1)
    cache3=cache13[:m*topk*k].view(m*topk,k)
    if selected['w2']['block_shape'][0]!=block:
        second=moe_align_block_size(local_ids,selected['w2']['block_shape'][0],layer.w2_post_scale.numel(),ignore_invalid_expert=True)
        common.update(sorted_ids=second[0],expert_ids=second[1],num_tokens_padded=second[2])
    HummingMethod.forward_layer(layer,inputs=cache2,outputs=cache3,top_k=1,sublayer_name='w2',tuning_config=configs['w2'],**common)
    cache3.mul_(layer.w2_post_scale[safe,None])
    return cache3.view(m,topk,k),local_ids


def main():
    p=argparse.ArgumentParser();p.add_argument('--sglang-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--hidden',type=int,default=512);p.add_argument('--experts',type=int,default=4);p.add_argument('--tokens',nargs='+',type=int,default=[1,8,33,128])
    p.add_argument('--import-only',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
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
    print('Preparing Humming canonical weights',flush=True)
    humm=prepare_humming(raw,torch)
    print('Preparing Marlin reference weights',flush=True)
    prepare_moe_nvfp4_layer_for_marlin(raw)
    method=MethodType(load_method(args.sglang_root,'baseline',torch)[0],raw)
    norm=torch.linspace(.5,1.5,args.hidden,device='cuda',dtype=torch.bfloat16)
    report={'runtime':runtime,'humming_version':'0.1.12','gpu':torch.cuda.get_device_name(),'rows':[],
            'component_gate':{'relative_l2_max':.003,'absolute_max':.03125},
            'scope':'component only; original BF16 global-scale output boundaries; no input quantization'}
    for m in args.tokens:
      for pattern in ('uniform','hot_rank','hot_expert','all_remote'):
        x=torch.randn((m,args.hidden),device='cuda',dtype=torch.bfloat16)/20
        ids=torch.tensor(make_routes(m,8,args.experts*8,8,pattern,216),device='cuda',dtype=torch.int32)
        scores=torch.rand((m,8),device='cuda');scores/=scores.sum(-1,keepdim=True)
        local=raw._expert_lookup[ids+1]
        def finish(v,li):return fused_post_expert_norm_weighted_sum(v,scores,norm_weight=norm,eps=1e-6,weight_first=True,slot_expert_ids=li)
        def baseline():return finish(*method(x,scores,ids))
        def candidate():return finish(*indexed(humm,x,local,torch))
        print(f'case m={m} pattern={pattern}',flush=True)
        b=baseline();a=candidate();torch.cuda.synchronize()
        diff=a.float()-b.float();rel=(diff.norm()/b.float().norm().clamp_min(1e-20)).item();ab=diff.abs().max().item()
        row={'m':m,'pattern':pattern,'relative_l2':rel,'max_abs':ab,'passed':bool(torch.isfinite(a).all()) and rel<=.003 and ab<=.03125}
        report['rows'].append(row);(args.out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
    report['passed']=all(r['passed'] for r in report['rows']);(args.out/'results.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()


