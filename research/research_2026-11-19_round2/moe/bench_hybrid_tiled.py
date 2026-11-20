"""Controlled complete-FFN cost of transient WGS->native fallback on local SM86.

This deliberately measures the fallback cost, not the unavailable H200 WGS decode
kernel. Native and WGS weight copies coexist only in this comparison fixture.
"""
import argparse
import ast
import gc
import hashlib
import json
import statistics
import sys
from pathlib import Path
import torch
here=Path(__file__).resolve().parent;root=here.parents[1]
for path in (root/'research_2026-11-18_target_moe',root/'swarm_2026-11-16/moe/humming_vendor',
             root/'swarm_2026-11-16/moe/phase2',root/'optimization_marlin_tiles_2026-11-15'):
    sys.path.insert(0,str(path))
import bench_experts as fixture
from kernel_runtime import bootstrap
bootstrap(root/'sglang')
from fixture_tuning_cache import cached_threshold_for_preparation
from hybrid_facade import SharedWeightWorkspace
import wgs_bridge
from wgs_bridge_tiled import convert as tiled_convert
original_convert=wgs_bridge.convert
pack=fixture.module('old_wgs_pack',root/'research_2026-11-18_target_moe/native_package_02/hopper_pack.py').pack_experts
ops=fixture.SOURCE/'kernels/ops/moe'
fixture.module('sglang.kernels.ops.moe.muse_spark_humming_scale',ops/'muse_spark_humming_scale.py')
fixture.module('sglang.kernels.ops.moe.muse_spark_humming_split',ops/'muse_spark_humming_split.py')
finish=fixture.module('bridge_parent_finish',ops/'muse_spark_post_expert_finalize_triton.py').fused_post_expert_norm_weighted_sum
path=fixture.SOURCE/'srt/layers/muse_spark/muse_spark_nvfp4_moe.py'
names={'_prepare_humming_experts','_humming_expert_outputs_default','_humming_expert_outputs'}
ns={'torch':torch};exec(compile(ast.Module(body=[n for n in ast.parse(path.read_text()).body if getattr(n,'name',None) in names],type_ignores=[]),str(path),'exec'),ns)
parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
parser.add_argument('--rows',type=int,nargs='+',default=[128,256,753,4097]);parser.add_argument('--pairs',type=int,default=6)
args=parser.parse_args();assert args.pairs>=4 and args.pairs%2==0
LIMIT=4*1024**3;e=32
report=dict(complete=False,seed=args.seed,device=torch.cuda.get_device_name(),cases=[],
    baseline_sha256=hashlib.sha256((fixture.RELEASE/'candidate.diff').read_bytes()).hexdigest(),
    bridge_sha256=hashlib.sha256((here/'wgs_bridge_tiled.py').read_bytes()).hexdigest(),
    scope='Tiled versus generic WGS restoration, both including identical complete W13/scale/SiLU/W2/finalizer. Parent correctness reference uses canonical weights. LocalSM86 only.',
    bench={'instances':1,'shapes':{}},maximum_preflight_live_bytes=LIMIT)
path=here/f'tiled_full_ffn_{args.seed}.json'
def save():path.write_text(json.dumps(report,indent=2),encoding='utf-8')
def guard(extra):
    live=torch.cuda.memory_allocated();free,total=torch.cuda.mem_get_info()
    if live+extra>LIMIT or free<extra+256*1024**2:raise RuntimeError(('GPU live/free guard',live,extra,free))
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    if available<1024**3:raise RuntimeError('host headroom guard')
def digest(raw,final,ids):
    a=hashlib.sha256();b=hashlib.sha256()
    for i in range(0,raw.size(0),64):
        a.update(raw[i:i+64][ids[i:i+64]>=0].view(torch.uint8).cpu().numpy().tobytes())
        b.update(final[i:i+64].view(torch.uint8).cpu().numpy().tobytes())
    return a.hexdigest(),b.hexdigest()
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    guard(3*e*12288*2304+64*1024**2)
    torch.manual_seed(args.seed);raw=torch.nn.Module();owner=torch.nn.Module();owner.experts=e
    for stage,n in (('w13',8192),('w2',4096)):
        w=torch.randint(256,(e,n,2048),device='cuda',dtype=torch.uint8)
        sf=torch.randint(33,70,(e,n,256),device='cuda',dtype=torch.uint8)
        gs=torch.linspace(.125,.5,e,device='cuda',dtype=torch.bfloat16)
        for suffix,value in [('weight',w),('weight_scale',sf.view(torch.float8_e4m3fn)),('weight_scale_2',gs)]:
            raw.register_parameter(stage+'_'+suffix,torch.nn.Parameter(value,requires_grad=False))
        owner.register_buffer(stage,pack(w,sf.view(torch.float8_e4m3fn),stage=='w13'))
        owner.register_buffer(stage+'_post_scale',gs)
    with cached_threshold_for_preparation(here):parent=ns['_prepare_humming_experts'](raw,torch)
    del raw,w,sf,gs,value
    workspace=SharedWeightWorkspace(e,owner.w13.device)
    workspace.prepare(owner)  # Reuses the already prepared native default tables.
    workspace.seal_for_capture()
    helper=ns['_humming_expert_outputs'];norm=torch.linspace(.7,1.2,4096,device='cuda',dtype=torch.bfloat16)
    report['owned_arena_bytes']=workspace.bytes
    for m in args.rows:
        graph_mode=m<=1536
        # Preflight all graph-private intermediate lifetimes, not just returned outputs.
        projected=(m*8*12288*2+m*4096*2+32*1024**2)*(3 if graph_mode else 1)
        guard(projected)
        x=torch.empty((m,4096),device='cuda',dtype=torch.bfloat16)
        ids=torch.empty((m,8),device='cuda',dtype=torch.int32)
        scores=torch.empty((m,8),device='cuda',dtype=torch.float32)
        def update(seed,mode):
            gen=torch.Generator(device='cuda').manual_seed(seed)
            x.normal_(std=.023,generator=gen);ids.fill_(-1)
            routes=torch.rand((m,256),device='cuda',generator=gen)
            if mode=='skew':routes[::3,:8]+=2
            chosen=routes.topk(8).indices
            ids.copy_(torch.where(chosen<32,chosen,-1))
            scores.uniform_(generator=gen);scores.div_(scores.sum(1,keepdim=True))
        def run(which,raw_output=False):
            if which=='parent':y,loc=helper(parent,x,ids,torch,defer_downscale=True,global_experts=256)
            else:
                wgs_bridge.convert=tiled_convert if which=='candidate' else original_convert
                y,loc=workspace.project(owner,helper,x,ids,defer_downscale=True,global_experts=256)
            z=finish(y,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=loc,expert_output_scale=owner.w2_post_scale)
            return (y,z) if raw_output else z
        for mode in ('uniform','skew'):
            update(args.seed+m,mode)
            a,z=run('parent',True);gold=digest(a,z,ids);del a,z
            a,z=run('candidate',True);actual=digest(a,z,ids);del a,z
            assert actual==gold,(m,mode,actual,gold)
            graphs={};outputs={}
            if graph_mode:
                for arm in ('baseline','candidate','control'):
                    run(arm);stream.synchronize();gc.collect();torch.cuda.empty_cache()
                    g=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g,stream=stream):outputs[arm]=run(arm)
                    graphs[arm]=g
                del g
            samples={arm:[] for arm in ('baseline','candidate','control')}
            for iteration in range(args.pairs):
                update(args.seed+m+iteration+1,mode)
                order=['baseline','candidate','control']
                if iteration%2:order.reverse()
                for arm in order:
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    begin.record()
                    if graph_mode:graphs[arm].replay()
                    else:run(arm)
                    end.record();end.synchronize();samples[arm].append(begin.elapsed_time(end)*1000.)
            row=dict(m=m,mode=mode,capture=graph_mode,correct=True,samples_us=samples,
                     median_speedup=statistics.median(samples['baseline'])/statistics.median(samples['candidate']))
            report['cases'].append(row)
            report['bench']['shapes'][f'm{m}_{mode}']={'baseline':samples['baseline'],'candidate':samples['candidate'],'control':samples['control']}
            save();print(json.dumps(row),flush=True)
            del graphs,outputs
            gc.collect();torch.cuda.empty_cache()
        del x,ids,scores
        gc.collect();torch.cuda.empty_cache()
    report.update(complete=True,correct=True,historical_peak_allocated=torch.cuda.max_memory_allocated(),historical_peak_reserved=torch.cuda.max_memory_reserved())
    assert report['historical_peak_allocated']<=LIMIT
    save();print(json.dumps({k:v for k,v in report.items() if k not in ('cases','bench')}),flush=True)
