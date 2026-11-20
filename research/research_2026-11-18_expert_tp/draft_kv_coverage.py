"""Audit DFlash draft-KV tail coverage at publication, branch, and resume boundaries."""
import json
import math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
W=2048
CHUNK=16384

def clipped_end_only(length,window,chunk):
    materialized=np.zeros(length,dtype=bool)
    counts=[]
    for begin in range(0,length,chunk):
        end=min(length,begin+chunk)
        keep=max(begin,length-window)
        n=max(0,end-keep)
        if n: materialized[keep:end]=True
        counts.append(n)
    return materialized,counts

def inspect_branch(valid,common,new,window,page):
    stop=common+new
    required_start=max(0,stop-window)
    required_end=min(stop,common)
    consumed=valid[required_start:required_end]
    missing=int((~consumed).sum())
    # The result of API REMATCH against a length-capped key may be shorter due
    # to target-side SWA tombstones, host eviction, locks, etc. Fresh slots
    # are required from that point; never overwrite the original shared branch.
    if missing:
        first_missing=required_start+int(np.flatnonzero(~consumed)[0])
        clipped=min(common,first_missing//page*page)
    else: clipped=common
    # Model distinct target re-extend slots with a fully materialized observed tail.
    replacement=np.concatenate([valid[:clipped],np.ones(stop-clipped,dtype=bool)])
    assert replacement[required_start:stop].all()
    return dict(common=common,new_tokens=new,missing=missing,
                maximum_allowed_target_hit=clipped,target_rows_recomputed=common-clipped)

def main():
    parent=json.loads((HERE.parent/"release_2026-11-18/scored_submission_status.json").read_bytes())["result"]
    shape=parent["shape"]; n=shape["first_turn"]; later=shape["new_per_turn"]
    valid,per_chunk=clipped_end_only(n,W,CHUNK)
    # Checkpoint a long prefix ending *inside* an earlier published chunk, then diverge.
    negative=inspect_branch(valid,32768,1,W,1)
    assert negative["missing"]==2047
    same=inspect_branch(valid,n,later,W,1)
    assert same["missing"]==0
    rng=np.random.default_rng(2048)
    for _ in range(5000):
        oldlen=int(rng.integers(5,500));win=int(rng.integers(1,100));common=int(rng.integers(oldlen+1))
        ext=int(rng.integers(1,150));page=int(rng.choice([1,4,8,16,64]))
        v=rng.random(oldlen)>.5
        inspect_branch(v,common,ext,win,page)
    # Hashed/physical IDs cannot be sufficient page-validity keys unless the
    # metadata also follows ownership on reclaim. Simulate stale positive entry.
    original_page_id=12
    last_generation={original_page_id:1}; recycled_generation=2
    assert original_page_id in last_generation and last_generation[original_page_id]!=recycled_generation

    target_active=42e9;tp=8
    fc=2*40960*8192
    layer_kv=2*8192*(2*128)*(5)
    local_flops=fc+layer_kv
    skipped=n-int(valid.sum())
    q=1-W/n
    result=dict(model_config="https://huggingface.co/RadixArk/Muse-Spark-Open-DFlash2/blob/main/config.json",
        raw_shape=shape,heads=dict(draft_attention_layers=5,window_including_current=W,window_left=W-1,
          target_taps=[1,16,30,44,59],conv_taps=2,conv_group=16,block_size=8),
        source_evidence=dict(worker="dflash_worker_v2.py:674,1821,2388,2408,2416,2984; includes immediate radix-safety note, shared addresses and aux capture lifetime",
          conv="models/dflash.py:520,536,830; K-tap applies inside fixed draft block; rowwise projected context is not convolved",
          hi_cache="hybrid_pool_assembler.py:1642-1665. Draft sidecar follows KV indices; receipt startup confirms per-rank host pool of 30.30 GB / 11,837,121 tokens",
          chunk="scheduler.py:3772; common.py:271; each extend_end can be published unless skip_radix_cache_insert",
          split="unified_radix_cache.py:613, unified_tree_core.py:1496; scored startup names RustUnifiedTreeCore",
          match="schedule_batch.py:_compute_max_prefix_len includes input_len-1; capping must precede the authoritative tree match/lock"),
        tests=dict(known_endpoint_materialized_by_chunk=per_chunk,KnownEndpointRows=int(valid.sum()),
          aligned_middle_branch=negative,same_tip_append=same,random_page_repair_cases=5000,
          scope="Token coverage only. Source guards rely on real fresh allocation, prefix locking and host transfer. Different target and draft GEMM shapes can change numerical association; requires hardware equivalence."),
        costs=dict(removed_rows=skipped,removed_fraction=q,joint_FC_KV_flops_per_token_rank=local_flops,
          fc_flops_per_token_rank=fc,layer_kv_flops_per_token_rank=layer_kv,
          removed_tflops_rank=skipped*local_flops/1e12,
          approximate_linear_flop_share_vs_target_plus_draft=local_flops/(2*target_active/tp+local_flops),
          if_equal_gemm_throughput_first_prefill_speed=1/(1-q*local_flops/(2*target_active/tp+local_flops)),
          if_no_coverage_metadata_unconditional_extra_target_rows_per_later= W-later,
          unconditional_later_recompute_flop_proxy_rank_12_turns=(shape["turns"]-1)*(W-later)*2*target_active/tp/1e12,
          per_request_raw_taps_gib=n*5*8192*2/2**30,per_request_projected_gib=n*8192*2/2**30,
          actual_sidecar_bytes_per_row_approx=30.30e9/11837121,
          fraction_of_cold_first_TTFT_required_for_draft_to_yield_first_speed={str(g):(1-1/g)/q for g in (1.2,1.35,1.5,1.8)},
          scope="42B published target active params approximate linear FLOPs (ignores attention, quantization, imbalance and LM head selectivity); no end-to-end or H200 stage latency measured."),
        sources=["https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4","https://huggingface.co/RadixArk/Muse-Spark-Open-DFlash2"])
    (HERE/"draft_kv_coverage.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(tests=result["tests"],costs=result["costs"])))
if __name__=="__main__":main()

