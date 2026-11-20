"""Admission and scratch for eager FA3 tree verification. All decisions precede target launch."""
import torch
from sglang.srt.mem_cache.kv_loc_plan import IdSpaceKind
from sglang.srt.runtime_context import get_parallel
from sglang.srt.layers.dp_attention import get_dp_tp_group
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
from sglang.srt.speculative.spec_utils import SIMULATE_ACC_LEN
from sglang.srt.speculative.fixed_tree_adapter import TreeStep, WIDTH


def tree_runtime_compatible(worker):
    runner=worker.model_runner
    backend=runner.attn_backend
    par=get_parallel()
    return (SIMULATE_ACC_LEN <= 0 and worker.lilicorr is None and not worker._need_mamba_verify_commit
        and not par.attn_dp_enabled and par.attn_dcp_size==1 and not par.enable_prefill_cp
        and type(backend).__name__=="FlashAttentionBackend"
        and type(runner.model).__name__ in ("MuseSparkTextForCausalLM","MuseSparkForConditionalGeneration")
        and backend.fa_impl_ver==3 and backend.page_size==1 and not backend.use_mla
        and not backend.fa_skip_kv_cache and not backend.kv_cache_is_mxfp8 and not backend.has_local_attention
        and backend.use_sliding_window_kv_pool and (backend.sliding_window_size or -1)>0
        and getattr(backend,"_kv_shard_pool",None) is None
        and getattr(runner.token_to_kv_pool,"swa_req_ring_size",None) is None)


def select_tree(worker, batch, prefix, original_slots, anchors, draft_graph):
    sampler = worker._draft_sampler
    if not draft_graph or getattr(sampler,"tree_scores",None) is None:
        return None
    if not tree_runtime_compatible(worker):
        return None
    runner=worker.model_runner
    backend=runner.attn_backend
    if (batch.forward_mode.is_idle() or not worker._can_skip_verify_logits(batch,batch.sampling_info)
        or (batch.sampling_info is not None and not batch.sampling_info.is_all_greedy)):
        return None
    translator = runner.kv_index_translator
    if translator.defer_read_translate:
        return None
    table = runner.req_to_token_pool.req_to_token
    if translator.is_translating:
        window_space = translator.space(IdSpaceKind.SLIDING_WINDOW)
        if window_space is None or window_space.read_v2p is None or translator.full_v2p_table is None:
            return None
        full,swa = translator.full_v2p_table,window_space.read_v2p
    else:
        swa = getattr(runner.token_to_kv_pool,"full_to_swa_index_mapping",None)
        if swa is None:
            return None
        identity = getattr(worker,"_fixed_tree_identity",None)
        if identity is None or identity.numel()!=swa.numel() or identity.device!=swa.device:
            identity = torch.arange(swa.numel(),dtype=torch.int32,device=swa.device)
            worker._fixed_tree_identity=identity
        full=identity
    bs = len(batch.reqs)
    if bs==0 or bs!=prefix.numel() or bs!=original_slots.shape[0]:
        return None
    # kv_allocated_len includes the complete reservation, including unsynchronized
    # over-allocation. A later bound on committed tokens would be insufficient.
    extent = max(r.kv.kv_allocated_len for r in batch.reqs)
    if not WIDTH<=extent<=table.shape[1]:
        return None
    cap = min(table.shape[1],1<<(int(extent)-1).bit_length())
    slab = getattr(worker,"_fixed_tree_slab",None)
    if slab is None or slab.batch<bs or slab.full.shape[1]<cap:
        if slab is not None:
            bs_cap,cap = max(slab.batch,bs),max(slab.full.shape[1],cap)
        else:
            bs_cap=bs
        slab=TreeStep(bs_cap,cap,table.device,backend.sliding_window_size)
        worker._fixed_tree_slab=slab
    step=slab.active(bs)
    step.propose(sampler.candidate_out[:bs],sampler.tree_scores[:bs],anchors.contiguous())
    # Keep the topology and the request-table permutation identical across ranks.
    # This is a single collective with one contiguous 160-byte payload per request.
    tp_group=get_dp_tp_group()
    if tp_group.world_size>1:
        tp_group.broadcast(step.wire,src=0)
    step.stage(table,batch.req_pool_indices,prefix,original_slots,full,swa,backend.sliding_window_size)
    return step


def accept_tree(worker,step,prediction,logits,batch,prefix,original_slots):
    if prediction is None:
        prediction=torch.argmax(logits,dim=-1)
    prediction=prediction.contiguous().view(step.batch,WIDTH)
    tp_group=get_dp_tp_group()
    if tp_group.world_size>1 and not worker._tp_sync.enabled(SpecTpSyncSite.DFLASH_ACCEPT_GREEDY):
        tp_group.broadcast(prediction,src=0)
    else:
        worker._tp_sync.sync(SpecTpSyncSite.DFLASH_ACCEPT_GREEDY,prediction)
    accept,step.counts,step.bonus,step.emitted,step.new_lens=worker._next_accept_bonus_buffers(step.batch)
    step.commit(prediction.contiguous(),worker.model_runner.req_to_token_pool.req_to_token,
        batch.req_pool_indices,prefix,original_slots)
    torch.sub(step.counts,1,out=accept)
    return accept,step.counts,step.bonus,step.emitted,step.new_lens

