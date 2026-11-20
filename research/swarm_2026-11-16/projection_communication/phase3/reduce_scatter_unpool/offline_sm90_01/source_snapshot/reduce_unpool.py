"""Isolated opt-in diagnostic. Not installed in scored or active model sources."""
import torch
from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit, make_cpp_args
from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import CustomAllReduceV2
from sglang.srt.environ import envs


def selected_policy(group, x, weight, *, model_eligible, experimental=False):
    # All ranks must pass the same model and metadata guards. No local routing decisions.
    if not experimental or not model_eligible or torch.compiler.is_compiling(): return None
    if envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get(): return None
    if type(group) is not GroupCoordinator or group.world_size not in (2,4,8): return None
    comm=group.ca_comm
    if type(comm) is not CustomAllReduceV2 or comm.disabled or comm.override_algo is not None: return None
    if not x.is_cuda or x.ndim!=2 or weight.ndim!=2: return None
    if x.dtype!=torch.bfloat16 or weight.dtype!=x.dtype or x.device!=weight.device: return None
    if not x.is_contiguous() or not weight.is_contiguous(): return None
    if torch.cuda.get_device_capability(x.device)!=(9,0): return None
    m,k=x.shape; n=weight.shape[0]
    if m<512 or m%group.world_size or weight.shape[1]!=k: return None
    if min(k,n)<1024 or k%128 or n%128: return None
    if x.data_ptr()%16 or weight.data_ptr()%16: return None
    if group._can_use_flashinfer_allreduce(x): return None
    if group._resolve_outplace_all_reduce_method(x)!='ca': return None
    config=comm._pick_config(x.numel()*x.element_size(),comm._can_use_graph())
    if config is None or config.algo!=AllReduceAlgo.TWO_SHOT_PULL: return None
    if config.use_graph and config.use_multicast: return None
    if not config.use_graph and x.numel()*x.element_size()>comm.max_pull_size: return None
    return config


@cache_once
def module(world_size):
    from pathlib import Path
    args=make_cpp_args(torch.bfloat16,world_size,is_arch_support_pdl())
    return load_jit('muse_reduce_scatter_rows_diagnostic',*args,
       cuda_files=[str(Path(__file__).with_name('reduce_scatter_rows.cuh'))],
       cuda_wrappers=[('run',f'MuseReduceScatterRows<{args}>::run')])


def try_reduce_unpool(group,x,weight,*,model_eligible,experimental=False):
    """None means caller must run its original AR + unpool path without mutation.

    model_eligible must retain the scored model guards: Humming/use_marlin,
    moe_tp_size==1, no SP/DP/CP, post projection exists, not already_reduced.
    This diagnostic API has no environment switch or installed model caller.
    """
    policy=selected_policy(group,x,weight,model_eligible=model_eligible,experimental=experimental)
    if policy is None: return None
    comm=group.ca_comm
    local=torch.empty((x.shape[0]//group.world_size,x.shape[1]),dtype=x.dtype,device=x.device)
    graph_params=comm._allocate_graph_row(x,x.numel()*x.element_size()) if policy.use_graph else None
    module(group.world_size).run(comm.obj,x,local,graph_params,policy.use_multicast)
    # Identical GEMM dimensions, FP32 accumulator output and BF16 cast to scored
    # prefill_row_projection. No K partition, padding, norm or expert changes.
    projected_fp32=torch.mm(local,weight.t(),out_dtype=torch.float32)
    projected=projected_fp32.to(x.dtype)
    del projected_fp32
    return group.all_gather(projected,dim=0)
