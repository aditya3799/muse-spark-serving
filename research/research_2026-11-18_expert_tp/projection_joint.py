"""CPU mathematical test of rank-local router/pooler projection and shared all-gather.
Each dot has a fixed reduction order. No CUDA algorithm parity or timing implied.
"""
import json
from pathlib import Path
import numpy as np

def bf16(a):
    bits = a.view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)

def widen(b):
    return (b.astype(np.uint32) << 16).view(np.float32)

def mm(x, w):
    return np.einsum("mk,nk->mn", x, w, optimize=False)

def pack(logits):
    # The gather's 16-bit all-zero Lamport marker is forbidden. Transmit a bitmask
    # for every original +0 halfword, so original negative zeros remain distinct.
    raw = np.ascontiguousarray(logits).view(np.uint16).reshape(len(logits), 64)
    zero = raw == 0
    ordinary = np.where(zero, 0x8000, raw).astype(np.uint16)
    masks = np.ones((len(logits),8), np.uint16)
    for i in range(64):
        masks[:,i//15] += zero[:,i].astype(np.uint16) << (i%15)
    return np.concatenate([ordinary,masks],axis=1)

def unpack(encoded):
    result = encoded[:,:64].copy()
    masks = encoded[:,64:]-np.uint16(1)
    for i in range(64):
        result[:,i] = np.where((masks[:,i//15] >> (i%15)) & 1, 0, result[:,i])
    return result.view(np.float32)

def route(logits,bias):
    scores=1/(1+np.exp(-logits))
    ids=np.argsort(-(scores+bias),kind="stable",axis=1)[:,:8]
    chosen=np.take_along_axis(scores,ids,axis=1)
    return ids,chosen/(np.sum(chosen,axis=1,keepdims=True)+1e-15)

def verify(seed,rows,k):
    rng=np.random.default_rng(seed)
    x=rng.standard_normal((rows,k),dtype=np.float32)
    activation=widen(bf16(x))
    weight=rng.standard_normal((256,k),dtype=np.float32)/np.float32(np.sqrt(k))
    pool_w=widen(bf16(rng.standard_normal((4096,k),dtype=np.float32)/np.float32(np.sqrt(k))))
    hi=widen(bf16(weight))
    lo=widen(bf16(weight-hi))
    target_pool=bf16(mm(activation,pool_w))
    target_router=mm(x,weight) if rows<=8 else mm(activation,hi)+mm(activation,lo)
    sharded=[]
    for rank in range(8):
        p=slice(rank*512,(rank+1)*512)
        e=slice(rank*32,(rank+1)*32)
        if rows<=8:
            local_router=mm(x,weight[e])
            local_pool=bf16(mm(activation,pool_w[p]))
        else:
            packed_weight=np.concatenate([pool_w[p],hi[e],lo[e]],axis=0)
            joint=mm(activation,packed_weight)
            local_pool=bf16(joint[:,:512])
            local_router=joint[:,512:544]+joint[:,544:]
        sharded.append(np.concatenate([local_pool,pack(local_router)],axis=1))
    send=np.stack(sharded,axis=1)
    gathered=np.where(send==0,0x8000,send).astype(np.uint16)
    got_pool=gathered[:,:,:512].reshape(rows,-1)
    ref_pool=np.where(target_pool==0,0x8000,target_pool)
    assert np.array_equal(got_pool,ref_pool)
    logits=np.concatenate([unpack(gathered[:,rank,512:]) for rank in range(8)],axis=1)
    assert np.array_equal(logits.view(np.uint32),target_router.view(np.uint32))
    bias=rng.standard_normal(256,dtype=np.float32)/16
    ids,weights=route(logits,bias)
    ref_ids,ref_weights=route(target_router,bias)
    assert np.array_equal(ref_ids,ids) and np.array_equal(ref_weights,weights)
    wrong=mm(activation,weight) if rows<=8 else mm(activation,hi)
    errors=int(np.count_nonzero(wrong!=target_router))
    assert errors
    return dict(seed=seed,rows=rows,k=k,correct_router_words=logits.size,correct_pooler_words=got_pool.size,
                expected_top8_ids=ids.size,wrong_input_or_dropped_split2_residual_changed_logits=errors)

def main():
    data=[]
    for seed,rows,k in [(193,1,8192),(194,8,8192),(195,32,512),(196,128,512)]:
        data.append(verify(seed,rows,k))
    rng=np.random.default_rng(349)
    raw=rng.integers(0,2**32,size=(128,32),dtype=np.uint32)
    raw[0,:8]=[0,0x80000000,0x7f800000,0xff800000,0x7fc00000,0x00008000,0x80008000,0xffffffff]
    packed=pack(raw.view(np.float32))
    assert packed.shape==(128,72) and not (packed==0).any()
    assert np.array_equal(unpack(packed).view(np.uint32),raw)
    result=dict(cases=data,all_bit_pattern_fixture=raw.size,local_pooler_words=512,local_logit_words=72,
                combined_rank_width=584,added_gather_traffic_fraction=72/512,
                scope="Fixed-order CPU GEMM algebra and exact transport. Real Hopper GEMM accumulation, route tie ordering, MPI/PDL protocol, and speed unmeasured.")
    Path(__file__).with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
if __name__=="__main__":
    main()

