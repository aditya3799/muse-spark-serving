"""Actual frozen Humming FFN via restored shared weights, with graph/layer reuse."""
import argparse
import ast
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch

here=Path(__file__).resolve().parent
root=here.parents[1]
for path in (root/'research_2026-11-18_target_moe',
             root/'swarm_2026-11-16/moe/humming_vendor',
             root/'swarm_2026-11-16/moe/phase2',
             root/'optimization_marlin_tiles_2026-11-15'):
    sys.path.insert(0,str(path))
import bench_experts as fixture
from kernel_runtime import bootstrap
bootstrap(root/'sglang')
from fixture_tuning_cache import cached_threshold_for_preparation
from hybrid_facade import SharedWeightWorkspace
pack=fixture.module('old_wgs_pack',root/'research_2026-11-18_target_moe/native_package_02/hopper_pack.py').pack_experts
ops=fixture.SOURCE/'kernels/ops/moe'
fixture.module('sglang.kernels.ops.moe.muse_spark_humming_scale',ops/'muse_spark_humming_scale.py')
fixture.module('sglang.kernels.ops.moe.muse_spark_humming_split',ops/'muse_spark_humming_split.py')
finish=fixture.module('bridge_parent_finish',ops/'muse_spark_post_expert_finalize_triton.py').fused_post_expert_norm_weighted_sum
path=fixture.SOURCE/'srt/layers/muse_spark/muse_spark_nvfp4_moe.py'
names={'_prepare_humming_experts','_humming_expert_outputs_default','_humming_expert_outputs'}
ns={'torch':torch}
exec(compile(ast.Module(body=[n for n in ast.parse(path.read_text()).body if getattr(n,'name',None) in names],type_ignores=[]),str(path),'exec'),ns)
parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=930239)
parser.add_argument('--rows',type=int,nargs='+',default=[8,128,256,753,4097])
args=parser.parse_args()
report=dict(complete=False,seed=args.seed,device=torch.cuda.get_device_name(),cases=[],metadata=[],
            scope='Original frozen BF16 full expert FFN+finalizer, synthetic actual-width weights; source-minimal local test, no H200 WGS kernel or normal CUTLASS execution')
output=here/f'facade_gpu_{args.seed}.json'
def save():output.write_text(json.dumps(report,indent=2),encoding='utf-8')
save()
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    workspace=SharedWeightWorkspace(2,torch.device('cuda',torch.cuda.current_device()))
    resident=[];parents=[]
    with cached_threshold_for_preparation(here):
        for layer_id in range(2):
            torch.manual_seed(args.seed+layer_id)
            raw=torch.nn.Module();owner=torch.nn.Module();owner.experts=2
            for stage,n in (('w13',8192),('w2',4096)):
                w=torch.randint(256,(2,n,2048),device='cuda',dtype=torch.uint8)
                sf=torch.randint(33,70,(2,n,256),device='cuda',dtype=torch.uint8)
                # Every finite signed E4M3 code, with original BF16 global-scale boundary.
                finite=torch.tensor([i for i in range(256) if i not in (127,255)],device='cuda',dtype=torch.uint8)
                sf[:,:254,0]=finite
                gs=torch.tensor([.125,.375],device='cuda',dtype=torch.bfloat16)
                for suffix,value in [('weight',w),('weight_scale',sf.view(torch.float8_e4m3fn)),('weight_scale_2',gs)]:
                    raw.register_parameter(stage+'_'+suffix,torch.nn.Parameter(value,requires_grad=False))
                owner.register_buffer(stage,pack(w,sf.view(torch.float8_e4m3fn),stage=='w13'))
                owner.register_buffer(stage+'_post_scale',gs)
            parent=ns['_prepare_humming_experts'](raw,torch)
            facade=workspace.prepare(owner)
            for stage in ('w13','w2'):
                assert dataclasses.asdict(parent.humming_metas[stage])==dataclasses.asdict(facade.humming_metas[stage])
                assert parent.fp32_configs[stage]==facade.fp32_configs[stage]
                assert getattr(parent,stage+'_post_scale').data_ptr()==getattr(facade,stage+'_post_scale').data_ptr()
            workspace.convert(owner)
            for stage in ('w13','w2'):
                meta=parent.humming_metas[stage]
                for attr in (meta.weight_name,meta.weight_scale_name):
                    a=getattr(parent,attr);b=getattr(facade,attr)
                    assert torch.equal(a.view(torch.uint8),b.view(torch.uint8)),(layer_id,attr)
            report['metadata'].append(dict(layer=layer_id,exact=True,layouts=workspace.layouts[facade],
                weight_storage_shared=True,post_scale_aliases_original=True))
            resident.append(owner);parents.append(parent)
            del raw,w,sf,gs,value,finite
    facade0=workspace.facades[resident[0]];facade1=workspace.facades[resident[1]]
    assert facade0.w13_weight.data_ptr()==facade1.w13_weight.data_ptr()
    assert facade0.w2_weight.data_ptr()==facade1.w2_weight.data_ptr()
    workspace.seal_for_capture()
    helper=ns['_humming_expert_outputs']
    norm=torch.linspace(.7,1.2,4096,device='cuda',dtype=torch.bfloat16)
    for m in args.rows:
        x=torch.empty((m,4096),device='cuda',dtype=torch.bfloat16)
        ids=torch.empty((m,8),device='cuda',dtype=torch.int32)
        scores=torch.empty((m,8),device='cuda',dtype=torch.float32)
        def inputs(seed,mode):
            generator=torch.Generator(device='cuda').manual_seed(seed)
            x.normal_(std=.023,generator=generator)
            ids.fill_(-1)
            if mode=='uniform':ids[:,0]=torch.arange(m,device='cuda')%2
            if mode=='skew':ids[:,0]=0;ids[::3,1]=1
            scores.uniform_(generator=generator);scores.div_(scores.sum(1,keepdim=True))
        def run(which):
            finals=[];raws=[]
            for li in (0,1,0):
                if which=='parent':
                    y,loc=helper(parents[li],x,ids,torch,defer_downscale=True,global_experts=256)
                else:
                    y,loc=workspace.project(resident[li],helper,x,ids,defer_downscale=True,global_experts=256)
                z=finish(y,scores,norm_weight=norm,eps=1e-8,weight_first=True,
                         slot_expert_ids=loc,expert_output_scale=parents[li].w2_post_scale)
                finals.append(z);raws.append(y)
            return raws,finals
        inputs(args.seed+m,'uniform')
        graphs={};outs={}
        for which in ('parent','candidate'):
            run(which);stream.synchronize()
            if m<=753:
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):outs[which]=run(which)
                graphs[which]=graph
        for iteration,mode in enumerate(('uniform','skew','remote')):
            inputs(args.seed+m+iteration+1,mode)
            if graphs:
                for graph in graphs.values():graph.replay()
            else:
                outs={which:run(which) for which in ('parent','candidate')}
            stream.synchronize()
            live=ids>=0;values=0
            for p,c in zip(outs['parent'][0],outs['candidate'][0]):
                assert torch.equal(p[live].view(torch.int16),c[live].view(torch.int16)),(m,mode,'raw')
                values+=int(live.sum())*4096
            for p,c in zip(outs['parent'][1],outs['candidate'][1]):
                assert torch.equal(p.view(torch.int16),c.view(torch.int16)),(m,mode,'final')
            row=dict(m=m,mode=mode,graph_replay=bool(graphs),sequence=[0,1,0],raw_values=values,final_values=3*m*4096,exact=True)
            report['cases'].append(row);save();print(json.dumps(row),flush=True)
        del graphs,outs,x,ids,scores
    try:workspace.close()
    except RuntimeError:report['graph_owned_close_rejected']=True
    else:raise AssertionError('graph workspace destroyed')
    report.update(complete=True,owned_weight_workspace_bytes=workspace.bytes,
                  historical_peak_allocated=torch.cuda.max_memory_allocated(),
                  historical_peak_reserved=torch.cuda.max_memory_reserved())
    save();print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)
