# SPDX-License-Identifier: Apache-2.0
"""Conservative startup qualification; original Humming remains the fallback.

The kernel uses original E2M1/E4M3 group16 coefficients, BF16 activations,
FP32 accumulators and original post-GEMM scales. Exact bit agreement is a
conservative activation gate, stronger than ordinary FP32-summation equivalence.
"""
import gc
import json
import logging
import statistics
import threading
import torch
from .muse_spark_grouped_build import library_path
from .muse_spark_grouped_runtime import Arena, MAX_M

LOG=logging.getLogger(__name__)
_BUILD=None
_STATE={}
MIN_M=4096
MAX_REQUEST_M=16384


def _meta_ok(layer):
    from humming import dtypes
    if layer.w13_post_scale.numel()!=32 or layer.w2_post_scale.numel()!=32:
        return False
    for stage,n in (('w13',8192),('w2',4096)):
        meta=layer.humming_metas[stage]
        if (meta.a_dtype!=dtypes.bfloat16 or meta.c_dtype!=dtypes.bfloat16 or
            meta.b_dtype!=dtypes.float4e2m1 or meta.bs_dtype!=dtypes.float8e4m3 or
            meta.weight_scale_group_size!=16 or meta.input_scale_group_size or
            meta.has_bias or meta.has_zero_point or meta.should_apply_bs_on_c or
            meta.use_fused_e8m0_scale or meta.shape_n!=n or meta.shape_k!=4096 or
            'wgmma' not in str(meta.mma_type).lower()):
            return False
        w=getattr(layer,meta.weight_name)
        sf=getattr(layer,meta.weight_scale_name)
        if (not w.is_contiguous() or not sf.is_contiguous() or
            w.numel()*w.element_size()!=32*n*4096//2 or
            sf.numel()*sf.element_size()!=32*n*4096//16):
            return False
    return True


def prepare_build(layer):
    global _BUILD
    if _BUILD is not None:
        return
    device=layer.w13_post_scale.device
    if device.type!='cuda' or torch.cuda.get_device_capability(device)!=(9,0):
        return
    if not _meta_ok(layer):
        return
    try:
        library_path()
        _BUILD=True
    except Exception as exc:
        _BUILD=False
        LOG.warning('Muse grouped build unavailable; original Humming retained: %s',exc)


def _difference(a,b):
    a=a.reshape(-1);b=b.reshape(-1)
    unequal=0;max_abs=0.;error2=0.;reference2=0.
    # Bound diagnostic scratch rather than materializing whole FP32 projections.
    for first in range(0,a.numel(),1<<20):
        av=a[first:first+(1<<20)];bv=b[first:first+(1<<20)]
        unequal+=int((av.view(torch.int16)!=bv.view(torch.int16)).sum())
        af=av.float();bf=bv.float();delta=af-bf
        max_abs=max(max_abs,float(delta.abs().max()))
        error2+=float(delta.square().sum());reference2+=float(af.square().sum())
    return dict(values=a.numel(),unequal_bits=unequal,max_abs=max_abs,
                relative_l2=(error2/max(reference2,1e-40))**.5)


def _raw_difference(a,b,ids):
    rows=[]
    for first in range(0,a.size(0),64):
        live=ids[first:first+64]>=0
        rows.append(_difference(a[first:first+64][live],b[first:first+64][live]))
    return dict(values=sum(x['values'] for x in rows),
                unequal_bits=sum(x['unequal_bits'] for x in rows),
                max_abs=max((x['max_abs'] for x in rows),default=0.),
                max_chunk_relative_l2=max((x['relative_l2'] for x in rows),default=0.))


def _inputs(device,m,local,seed):
    generator=torch.Generator(device=device).manual_seed(seed)
    x=torch.empty((m,4096),device=device,dtype=torch.bfloat16).normal_(std=.017,generator=generator)
    ids=torch.full((m,8),-1,device=device,dtype=torch.int32)
    if local:
        rows=torch.arange(m,device=device,dtype=torch.int32)
        for slot in range(local):
            ids[:,slot]=(rows*local+slot)%32
    return x,ids


def _checkpoint_difference(host,candidate,ids=None):
    records=[]
    for first in range(0,candidate.size(0),64):
        gold=host[first:first+64].to(candidate.device)
        actual=candidate[first:first+64]
        if ids is not None:
            live=ids[first:first+64]>=0
            gold=gold[live];actual=actual[live]
        records.append(_difference(gold,actual))
    return dict(values=sum(x['values'] for x in records),
                unequal_bits=sum(x['unequal_bits'] for x in records),
                max_abs=max((x['max_abs'] for x in records),default=0.),
                max_chunk_relative_l2=max((x['relative_l2'] for x in records),default=0.))


def _memory(device):
    free,total=torch.cuda.mem_get_info(device)
    return dict(allocated=torch.cuda.memory_allocated(device),
                reserved=torch.cuda.memory_reserved(device),
                historical_peak_allocated=torch.cuda.max_memory_allocated(device),
                historical_peak_reserved=torch.cuda.max_memory_reserved(device),
                cuda_free=free,cuda_total=total)


def _release_cache(device):
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()


def _times(fn):
    fn();fn()
    values=[]
    for iteration in range(6):
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        begin.record();fn();end.record();end.synchronize()
        values.append(begin.elapsed_time(end)*1000.)
    return values


def _qualify(reference,layer,device):
    from .muse_spark_post_expert_finalize_triton import fused_post_expert_norm_weighted_sum as finish
    memory={'before':_memory(device)}
    norm=torch.ones(4096,device=device,dtype=torch.bfloat16)
    checkpoints=[]
    # Native reference temporaries never coexist with the candidate arena.
    # Each checkpoint retains CPU BF16 only; input generation uses isolated seeds.
    for m,local,seed in ((4096,1,917411),(4097,1,917413),(4096,4,917419),(256,0,917421)):
        x,ids=_inputs(device,m,local,seed)
        scores=torch.full((m,8),.125,device=device,dtype=torch.float32)
        gold,_=reference(layer,x,ids,torch,retain=False,defer_downscale=True,global_experts=256)
        final=finish(gold,scores,norm_weight=norm,eps=1e-8,weight_first=True,
                     slot_expert_ids=ids,expert_output_scale=layer.w2_post_scale)
        checkpoints.append((m,local,seed,gold.cpu(),final.cpu()))
        del x,ids,scores,gold,final
    x,ids=_inputs(device,4096,1,917431)
    parent=_times(lambda:reference(layer,x,ids,torch,retain=False,defer_downscale=True,global_experts=256))
    del x,ids
    memory['native_before_arena']=_memory(device)
    _release_cache(device)
    arena=Arena(device)
    memory['arena_allocated']=_memory(device)
    records=[];exact=True
    for m,local,seed,gold_cpu,final_cpu in checkpoints:
        x,ids=_inputs(device,m,local,seed)
        scores=torch.full((m,8),.125,device=device,dtype=torch.float32)
        candidate,_=arena.forward(layer,x,ids,True)
        final=finish(candidate,scores,norm_weight=norm,eps=1e-8,weight_first=True,
                     slot_expert_ids=ids,expert_output_scale=layer.w2_post_scale)
        raw=_checkpoint_difference(gold_cpu,candidate,ids)
        final_diff=_checkpoint_difference(final_cpu,final)
        exact=exact and raw['unequal_bits']==0 and final_diff['unequal_bits']==0
        records.append(dict(m=m,local_slots=local,seed=seed,raw=raw,final=final_diff))
        del x,ids,scores,candidate,final
    x,ids=_inputs(device,4096,1,917431)
    candidate_times=_times(lambda:arena.forward(layer,x,ids,True))
    del x,ids,checkpoints,gold_cpu,final_cpu
    memory['candidate_before_release']=_memory(device)
    owned_bytes=arena.owned_bytes
    packed_bytes=sum(t.numel() for t in arena.packed)
    arena.close();del arena
    _release_cache(device)
    # Post-candidate native control catches gross clock/thermal drift without
    # retaining an arena during the native reference's large temporary GEMMs.
    x,ids=_inputs(device,4096,1,917431)
    control=_times(lambda:reference(layer,x,ids,torch,retain=False,defer_downscale=True,global_experts=256))
    del x,ids,norm
    memory['native_control_after_release']=_memory(device)
    _release_cache(device)
    parent_median=statistics.median(parent);control_median=statistics.median(control)
    gain=min(parent_median,control_median)/statistics.median(candidate_times)
    drift=abs(parent_median-control_median)/min(parent_median,control_median)
    enabled=bool(exact and gain>1.10 and drift<=.05)
    result=Arena(device) if enabled else False
    memory['after_qualification']=_memory(device)
    LOG.info('Muse grouped qualification %s',json.dumps(dict(device=str(device),
        dtype='BF16/BF16, E2M1+E4M3 group16, FP32 accumulation, original post scales',
        token_tile=64,chunk_rows=MAX_M,minimum_request_rows=MIN_M,
        qualification_scope='first actual layer, finite synthetic inputs; not universal parity',
        timing_scope='separate parent/candidate/control epochs; conservative startup filter, not paired promotion',
        original_weight_storage_retained=True,packed_bytes=packed_bytes,
        owned_scratch_bytes=owned_bytes,memory=memory,
        records=records,times_us=dict(parent=parent,candidate=candidate_times,control=control),
        synthetic_gain=gain,parent_control_drift=drift,exact=exact,enabled=enabled)))
    return result


def _serialized_execution():
    from sglang.srt.layers.moe.utils import is_tbo_enabled,is_sbo_enabled
    from sglang.srt.distributed.parallel_state import get_pipeline_model_parallel_world_size
    return not is_tbo_enabled() and not is_sbo_enabled() and get_pipeline_model_parallel_world_size()==1


def _stream_allowed(arena):
    from sglang.srt.runtime_context import get_resources
    if threading.get_ident()!=arena.thread_id:
        return False
    current=torch.cuda.current_stream(arena.device)
    capture=get_resources().streams.get('cuda_graph_capture')
    if capture is not None and current==capture:
        # GroupCoordinator.graph_capture waits for the warmup stream before entering;
        # runner_utils/pool.py explicitly guarantees prefill/decode replay serialization.
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    if current==torch.cuda.default_stream(arena.device):
        return arena.forward_stream is None
    if arena.forward_stream is None:
        # One-time transition from initialization/capture to the scheduler forward stream.
        torch.cuda.synchronize(arena.device)
        arena.forward_stream=current
        LOG.info('Muse grouped bound scheduler stream %s',current.cuda_stream)
    return current==arena.forward_stream


def try_grouped(reference,layer,x,ids,torch_module,retain=False,*,defer_downscale=False,global_experts=None):
    if (_BUILD is not True or retain or global_experts!=256 or x.ndim!=2 or
        not MIN_M<=x.size(0)<=MAX_REQUEST_M or x.size(1)!=4096 or
        x.dtype!=torch.bfloat16 or ids.dtype!=torch.int32 or ids.shape!=(x.size(0),8) or
        not x.is_cuda or ids.device!=x.device or not x.is_contiguous() or not ids.is_contiguous() or
        not _meta_ok(layer) or not _serialized_execution()):
        return None
    index=x.device.index
    if index not in _STATE:
        if torch.cuda.is_current_stream_capturing():
            return None
        # No fallback after a device execution error: a damaged CUDA context must fail loudly.
        _STATE[index]=_qualify(reference,layer,x.device)
        if _STATE[index] is False:
            return None
    arena=_STATE[index]
    if arena is False or not _stream_allowed(arena):
        return None
    if torch.cuda.is_current_stream_capturing():
        LOG.info("Muse grouped captured activation M=%s stream=%s",x.size(0),torch.cuda.current_stream(x.device).cuda_stream)
    return arena.forward(layer,x,ids,defer_downscale)
