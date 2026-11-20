"""Research-only producer policy: preserve all scored geometry and arithmetic fields."""
from copy import deepcopy

def producer_config(original, policy, sm_version=90):
    if sm_version != 90 or policy not in ('tma_b','tma_b_only','warp_spec','both','warp_tma_b_only'):
        return None
    if original.get('use_stream_k') is not False or original.get('use_f16_accum') is not False:
        return None
    block=original.get('block_shape',());warp=original.get('warp_shape',())
    if len(block)!=3 or len(warp)!=3 or any(not isinstance(x,int) or isinstance(x,bool) or x<=0 for x in (*block,*warp)):
        return None
    stages=original.get('num_stages',0)
    if not isinstance(stages,int) or isinstance(stages,bool) or stages<3:
        return None
    if any(b%w for b,w in zip(block,warp)):
        return None
    threads=32
    for b,w in zip(block,warp):threads*=b//w
    ws=policy in ('warp_spec','both','warp_tma_b_only');tma=policy in ('tma_b','tma_b_only','both','warp_tma_b_only')
    tma_scales=policy in ('tma_b','both')
    if threads+(128 if ws else 0)>1024 or (ws and (threads%128 or warp[2]//16<2)):
        return None
    result=deepcopy(original)
    result.update(use_warp_spec=ws,use_tma=tma,use_mbarrier=True,use_cp_async=True,
                  use_tma_a=False,use_tma_as=False,use_tma_c=False,
                  use_tma_b=tma,use_tma_bs=tma_scales,use_tma_bs2=False,
                  use_tma_bzp=False,use_tma_bias=False,
                  multi_cast_size_a=1,multi_cast_size_b=1)
    # No changes to block/warp shape, K partition, pipeline depth, stream-K,
    # FP32 accumulator, routing algorithm, scale/finalizer or persistent weights.
    immutable=('block_shape','warp_shape','num_stages','use_stream_k','use_f16_accum')
    assert all(result[k]==original[k] for k in immutable)
    return result
