"""Exact route placement and scratch bounds for token-aligned local route buckets."""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def placement(ids,capacity):
    topk=ids.shape[1]
    assert capacity>=topk
    active=ids>=0
    counts=active.sum(axis=1,dtype=np.int64)
    before=np.cumsum(counts)-counts
    stride=capacity-(topk-1)
    groups=before//stride
    local_slot=np.cumsum(active,axis=1)-1
    offsets=np.full_like(ids,-1,dtype=np.int64)
    for g in np.unique(groups):
        same=groups==g
        within=(before[same]-g*stride)[:,None]+local_slot[same]
        offsets[same]=np.where(active[same],within,-1)
        j=offsets[same][active[same]]
        if len(j):
            assert j[0]<=topk-1 and np.array_equal(j,np.arange(j[0],j[0]+len(j))) and j[-1]<capacity
    assert groups.max()<(ids.size+stride-1)//stride
    return groups, offsets

def make_ids(rng,m,profile):
    tags=rng.random((m,256),dtype=np.float32)
    if profile=="cluster":tags[::2,:8]+=2
    chosen=np.argpartition(tags,-8,axis=1)[:,-8:]
    if profile=="all_local":return np.tile(np.arange(8),(m,1))
    if profile=="all_remote":return np.full((m,8),-1)
    return np.where(chosen<32,chosen,-1)

def bf16(arr):
    x=np.array(arr,dtype=np.float32,order="C")
    b=x.view(np.uint32)
    return ((b+32767+((b>>16)&1))&np.uint32(0xffff0000)).view(np.float32)

def finalize(raw, ids, gate, norm):
    # Preserve original slot order; intermediate arrays represent scored BF16 rounding boundaries.
    m,k,h=raw.shape
    scale=np.linspace(.125,.5,32,dtype=np.float32)
    acc=np.zeros((m,h),np.float32)
    for j in range(k):
        live=ids[:,j]>=0
        y=bf16(raw[live,j]*scale[ids[live,j],None])*norm
        v=y/np.sqrt(np.mean(y*y,axis=1,keepdims=True)+1e-8)
        acc[live]+=v*gate[live,j,None]
    return bf16(acc)

def trial(m,profile,seed,check_math):
    rng=np.random.default_rng(seed)
    ids=make_ids(rng,m,profile)
    cap=2*m+7
    groups,off=placement(ids,cap)
    slots=ids>=0
    n_buckets=(ids.size+2*m-1)//(2*m)
    loads=[int(slots[groups==g].sum()) for g in range(n_buckets)]
    assert sum(loads)==int(slots.sum()) and max(loads)<=cap
    if check_math:
        h=47
        x=bf16(rng.normal(size=(m,h)));w=bf16(rng.normal(size=(32,h)))
        # Stand-in for W13/W2 results, deterministic and sensitive to row, expert, lane, and slot mapping.
        raw=bf16(x[:,None,:]*w[ids])
        weights=rng.random((m,8),dtype=np.float32)
        norm=bf16(np.linspace(.7,1.3,h,dtype=np.float32))
        want=finalize(raw,ids,weights,norm)
        got=np.empty_like(want)
        for g in range(n_buckets):
            mask=groups==g
            n=int(np.sum(mask))
            if not n:continue
            v=slots[mask]
            compact=np.empty((cap,h),np.float32)
            rows=np.nonzero(mask)[0]
            rr,ss=np.nonzero(v)
            compact[off[rows[rr],ss]]=bf16(x[rows[rr]]*w[ids[rows[rr],ss]])
            gathered=np.zeros((n,8,h),np.float32)
            gathered[v]=compact[off[mask][v]]
            got[mask]=finalize(gathered,ids[mask],weights[mask],norm)
        assert np.array_equal(want,got)
    unique=int(len(np.unique(ids[slots])))
    per_bucket=[int(len(np.unique(ids[(groups[:,None]==g)&slots]))) for g in range(n_buckets)]
    return dict(rows=m,profile=profile,seed=seed,buckets=n_buckets,capacity=cap,counts=loads,
                unique_experts_whole=unique,unique_experts_per_bucket=per_bucket,
                bucket_expert_touch_ratio=sum(per_bucket)/unique if unique else None,
                nonempty_buckets=sum(n>0 for n in loads),baseline_internal_bytes=m*8*4096*6,
                packed_internal_bytes=cap*4096*6,packed_input_bytes=cap*4096*2,
                packed_workspace_including_input_over_baseline=(cap*4096*8)/(m*8*4096*6),
                cpu_stand_in_checked=check_math)

def main():
    results=[trial(m,p,s,m<=753) for m in (8,32,128,256,753,16384)
             for p in ("uniform","cluster","all_local","all_remote") for s in (317,743)]
    assert all(len(r["counts"])==4 for r in results)
    report=dict(scope="CPU graph-shape and route ownership only. Fixed four buckets, no CPU query of live count. Existing Humming already skips invalid GEMM tiles. Budget includes W13/W2 aliases and gate, excluding scratch required by aligner, finalizer, maps and original inputs. Native GPU performance, exact actual finalizer parity and empty bucket handling remain unverified.",
                  rows=results)
    (HERE/"compact_plan.json").write_text(json.dumps(report,indent=2))
    print(len(results),[r["counts"] for r in results if r["rows"]==16384 and r["seed"]==317])
if __name__=="__main__":main()

