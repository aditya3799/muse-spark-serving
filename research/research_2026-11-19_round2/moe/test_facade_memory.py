"""Single-layer E32 correctness and phase-separated memory proof. No speed claims."""
import ast
import gc
import hashlib
import json
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
pack=fixture.module('old_wgs_pack',root/'research_2026-11-18_target_moe/native_package_02/hopper_pack.py').pack_experts
ops=fixture.SOURCE/'kernels/ops/moe'
fixture.module('sglang.kernels.ops.moe.muse_spark_humming_scale',ops/'muse_spark_humming_scale.py')
fixture.module('sglang.kernels.ops.moe.muse_spark_humming_split',ops/'muse_spark_humming_split.py')
finish=fixture.module('bridge_parent_finish',ops/'muse_spark_post_expert_finalize_triton.py').fused_post_expert_norm_weighted_sum
path=fixture.SOURCE/'srt/layers/muse_spark/muse_spark_nvfp4_moe.py'
names={'_prepare_humming_experts','_humming_expert_outputs_default','_humming_expert_outputs'}
ns={'torch':torch};exec(compile(ast.Module(body=[n for n in ast.parse(path.read_text()).body if getattr(n,'name',None) in names],type_ignores=[]),str(path),'exec'),ns)
helper=ns['_humming_expert_outputs'];seed=930257;m=4097;e=32
LIMIT=4*1024**3
report=dict(complete=False,seed=seed,rows=m,experts=e,device=torch.cuda.get_device_name(),
            maximum_preflight_live_bytes=LIMIT,preflight=[],phases={},correctness=[],
            scope='Single-layer synthetic full32-expert native fallback; phase-separated weights and bounded64-row CPU SHA256 of live raw/all final BF16 bytes; standalone peak counters only')
def save(): (here/'facade_memory.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
def state():
    free,total=torch.cuda.mem_get_info()
    return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                free=free,total=total)
def guard(extra,label):
    current=state();planned=current['allocated']+extra
    report['preflight'].append(dict(label=label,additional_bound=extra,planned_live=planned,current=current));save()
    if planned>LIMIT or current['free']<extra+256*1024**2:
        raise RuntimeError('preflight live/free GPU guard rejected '+label)
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    if available<1400*1024**2:raise RuntimeError('host memory headroom rejected '+label)
def release():
    torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
def output_digest(raw,final,ids):
    raw_hash=hashlib.sha256();final_hash=hashlib.sha256();raw_values=0
    for first in range(0,raw.size(0),64):
        live=ids[first:first+64]>=0
        chunk=raw[first:first+64][live]
        raw_values+=chunk.numel()
        raw_hash.update(chunk.view(torch.uint8).cpu().numpy().tobytes())
        final_hash.update(final[first:first+64].view(torch.uint8).cpu().numpy().tobytes())
        del chunk,live
    return dict(raw_sha256=raw_hash.hexdigest(),final_sha256=final_hash.hexdigest(),
                raw_values=raw_values,final_values=final.numel())
def inputs(mode):
    gen=torch.Generator(device='cuda').manual_seed(seed+{'uniform':1,'skew':2,'remote':3}[mode])
    x=torch.empty((m,4096),device='cuda',dtype=torch.bfloat16).normal_(std=.023,generator=gen)
    ids=torch.full((m,8),-1,device='cuda',dtype=torch.int32)
    if mode=='uniform':ids[:,0]=torch.arange(m,device='cuda')%e
    if mode=='skew':ids[:,0]=0;ids[::3,1]=17
    scores=torch.rand((m,8),device='cuda',generator=gen);scores.div_(scores.sum(1,keepdim=True))
    return x,ids,scores
norm=torch.linspace(.7,1.2,4096,device='cuda',dtype=torch.bfloat16)
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    guard(3*e*12288*2304+64*1024**2,'construct raw+native+WGS weights')
    torch.manual_seed(seed);raw=torch.nn.Module();owner=torch.nn.Module();owner.experts=e
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
    host_owner={name:getattr(owner,name).cpu() for name in ('w13','w2')}
    post_scales={name:getattr(owner,name) for name in ('w13_post_scale','w2_post_scale')}
    del owner
    release()
    # Standalone process: no serving global counters exist or are reset here.
    torch.cuda.reset_peak_memory_stats()
    report['phases']['native_start']=state()
    gold={}
    for mode in ('uniform','skew','remote'):
        guard(1100*1024**2,'native complete FFN '+mode)
        x,ids,scores=inputs(mode)
        y,loc=helper(parent,x,ids,torch,defer_downscale=True,global_experts=256)
        final=finish(y,scores,norm_weight=norm,eps=1e-8,weight_first=True,
                     slot_expert_ids=loc,expert_output_scale=parent.w2_post_scale)
        # Bounded64-row CPU digest; no complete native output survives this phase.
        gold[mode]=output_digest(y,final,ids)
        del x,ids,scores,y,loc,final
    report['phases']['native_peak']=state();del parent;release()
    report['phases']['after_native_release']=state()
    assert torch.cuda.memory_allocated()<32*1024**2
    guard(2*e*12288*2304+64*1024**2,'WGS resident plus shared restoration arena')
    owner=torch.nn.Module();owner.experts=e
    for name,tensor in host_owner.items():owner.register_buffer(name,tensor.cuda())
    for name,tensor in post_scales.items():owner.register_buffer(name,tensor)
    del host_owner,tensor
    workspace=SharedWeightWorkspace(e,owner.w13.device)
    facade=workspace.prepare(owner)  # Native default tables were already prepared above.
    workspace.seal_for_capture();release()
    torch.cuda.reset_peak_memory_stats()
    report['phases']['candidate_start']=state()
    for mode in ('uniform','skew','remote'):
        guard(1100*1024**2,'candidate complete FFN '+mode)
        x,ids,scores=inputs(mode)
        y,loc=workspace.project(owner,helper,x,ids,defer_downscale=True,global_experts=256)
        final=finish(y,scores,norm_weight=norm,eps=1e-8,weight_first=True,
                     slot_expert_ids=loc,expert_output_scale=owner.w2_post_scale)
        observed=output_digest(y,final,ids)
        assert observed==gold[mode],(mode,observed,gold[mode])
        report['correctness'].append(dict(mode=mode,**observed,digests_match=True))
        del x,ids,scores,y,loc,final
        save()
    report['phases']['candidate_peak']=state()
    assert report['phases']['candidate_peak']['peak_allocated']<=LIMIT
    report.update(complete=True,owned_arena_bytes=workspace.bytes,
                  observed_peak_allocated_delta=report['phases']['candidate_peak']['peak_allocated']-report['phases']['native_peak']['peak_allocated'],
                  observed_peak_reserved_delta=report['phases']['candidate_peak']['peak_reserved']-report['phases']['native_peak']['peak_reserved'],
                  source_sha256=hashlib.sha256((here/'hybrid_facade.py').read_bytes()).hexdigest())
    save();print(json.dumps(report),flush=True)
