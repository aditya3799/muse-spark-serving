"""CPU screen for output-feature-sharded NVFP4 expert TP. Throughput is unmeasured."""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
SERVING = HERE.parent
EXPERTS, TOPK, WIDTH = 256, 8, 4096
PAYLOAD = 3 * WIDTH**2 * (0.5 + 1/16)

def sample(m, seed, dist):
    rng = np.random.default_rng(seed)
    if dist == "normal":
        offset = .03 * rng.standard_normal(EXPERTS) / np.sqrt(-np.log(np.maximum(rng.random(EXPERTS), 1e-12)))
    else:
        offset = 4 * rng.random(EXPERTS)**6
    counts = np.zeros(EXPERTS, dtype="int64")
    for first in range(0, m, 256):
        values = rng.random((min(256, m-first), EXPERTS)) + offset
        ids = np.argpartition(values, -TOPK, axis=1)[:, -TOPK:]
        counts += np.bincount(ids.ravel(), minlength=EXPERTS)
    return counts

def distribution():
    rows = []
    for m in (8,32,128,256,753,16384):
        seeds = 192 if m <= 256 else 48 if m<=753 else 12
        for dist in ("normal","skewed"):
            outcomes = {p: [] for p in (2,4,8)}
            for seed in range(seeds):
                count = sample(m, 3773+seed*379, dist)
                tiles = (count+15)//16
                def stat(array,p):
                    return float(array.reshape(8//p,-1).sum(axis=1).max())/p
                for p in outcomes:
                    # Equal weight storage per GPU for each expert topology. Upper envelope of work on the busiest rank.
                    # The named bandwidth counters count unique active weights once; actual cache re-reads and tiling may differ.
                    touched = stat(count>0,p)
                    pairs = stat(count,p)
                    group_pairs = int(round(p*pairs))
                    payload_bytes = group_pairs*WIDTH*2*(p-1)/p
                    # Lower bound on per-rank received BF16 hidden, before W2. Excludes topology/final norm exchange.
                    outcomes[p].append([stat(count>0,1)/touched,stat(count,1)/pairs,stat(tiles,1)/stat(tiles,p),
                                        payload_bytes,stat(count>0,1),touched])
            for p, data in outcomes.items():
                a=np.array(data)
                rows.append(dict(rows=m, routing=dist, shard=p,seeds=seeds,
                                 median_weight_ratio=float(np.median(a[:,0])),weight_ratio_p10=float(np.quantile(a[:,0],.10)),
                                 median_route_ratio=float(np.median(a[:,1])),median_m16_tile_ratio=float(np.median(a[:,2])),
                                 median_receive_bytes=float(np.median(a[:,3])),median_critical_experts_ep8=float(np.median(a[:,4])),
                                 median_critical_expert_equivalents_sharded=float(np.median(a[:,5]))))
    return rows

def bf16(x):
    a=np.ascontiguousarray(x,dtype=np.float32)
    bits=a.view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)

def arithmetic():
    # Condensed 16-expert fixture with the identical group-size 16 and top-8 dimensions.
    # Weights use E2M1 representable codes times exactly representable E4M3 scales.
    device_results=[]
    for m,seed in ((1,5),(8,97),(32,149)):
        rng=np.random.default_rng(seed)
        e,h,i=16,128,128
        grid=np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],dtype=np.float32)
        def weights(n,k):
            codes=grid[rng.integers(0,16,(e,n,k))]
            scales=rng.choice(np.array([.0625,.125,.25,.375,.5],dtype=np.float32),(e,n,k//16))
            return codes*scales.repeat(16,axis=-1)
        gate,up,down=(weights(i,h),weights(i,h),weights(h,i))
        gscale=bf16(rng.uniform(.1,.8,e))
        dscale=bf16(rng.uniform(.1,.8,e))
        norm=bf16(rng.uniform(.7,1.2,h))
        ids=np.argsort(rng.random((m,e)),axis=1)[:,-8:]
        scores=rng.uniform(.1,2,(m,8)).astype(np.float32)
        scores/=scores.sum(axis=-1,keepdims=True)
        x=bf16(rng.normal(size=(m,h))*.1)
        def predict(p):
            raw=np.empty((m,8,h),dtype=np.float32)
            for expert in range(e):
                token,slot=np.where(ids==expert)
                if not token.size: continue
                chunks=[]
                for b in range(p):
                    sl=slice(b*i//p,(b+1)*i//p)
                    g=bf16(bf16(x[token]@gate[expert,sl].T)*gscale[expert])
                    u=bf16(bf16(x[token]@up[expert,sl].T)*gscale[expert])
                    chunks.append(bf16(g/(1+np.exp(-g))*u))
                hidden=np.concatenate(chunks,axis=1)
                parts=[]
                squares=[]
                for b in range(p):
                    sl=slice(b*h//p,(b+1)*h//p)
                    part=bf16(bf16(hidden@down[expert,sl].T)*dscale[expert])*norm[sl]
                    parts.append(part)
                    squares.append((part**2).sum(axis=-1))
                inv=1/np.sqrt(sum(squares)/h+1e-8)
                raw[token,slot]=np.concatenate(parts,axis=1)*inv[:,None]
            return bf16((raw*scores[:,:,None]).sum(axis=1))
        ref=predict(1)
        for p in (2,4,8):
            candidate=predict(p)
            rel=float(np.linalg.norm(candidate-ref)/np.linalg.norm(ref))
            device_results.append(dict(m=m,seed=seed,p=p,rel_l2=rel,exact=float((candidate==ref).mean())))
            assert rel<.004 and np.isfinite(candidate).all()
    return device_results

def main():
    scored=json.loads((SERVING/"release_2026-11-18/scored_submission_status.json").read_text())
    benchmark=scored["result"]
    fraction=1-benchmark["frontier"]["score"]/1.8
    budgets=[dict(users=r["concurrency"],decode_cycle_proxy_ms=r["acceptance"]/r["tok_s_per_user_p50"]*1000,
                  cycle_saving_proxy_ms_at_uniform_gain=fraction*r["acceptance"]/r["tok_s_per_user_p50"]*1000)
             for r in benchmark["rungs"]]
    result=dict(method="FP32 numpy CPU numerical fixture and paired analytic route-work simulation using reference.py's probability equations; independent RNG and synthetic routing, no hardware timing.",
                parent=scored["id"],required_equal_time_fraction=fraction,decode_only_budget=budgets,
                source="muse_moe_w4a16/reference.py lines 65-73",fixture=arithmetic(),model=distribution())
    (HERE/"result.json").write_text(json.dumps(result,indent=2)+"\n")

if __name__=="__main__":
    main()
