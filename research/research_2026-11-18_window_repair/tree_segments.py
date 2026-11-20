"""Request-disjoint replay for bounded causal branch segments and a serial-page oracle."""
import collections
import heapq
import json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;K=H.parent/"research_2026-11-18_rl/lattice_tree"

def proposal(ids,logp,anchor,cap):
    tokens=[int(anchor)];parents=[-1];depth=[0];choice=[0];value=[0.]
    pending=[];child_counts=[0];leaves=1
    for node in range(7):
        d=depth[node]
        if d<7:
            for k,lp in enumerate(logp[d,choice[node]]):
                heapq.heappush(pending,(-float(np.float32(value[node]+lp)),node,k))
        while True:
            neg,parent,k=heapq.heappop(pending)
            if child_counts[parent] and leaves>=cap:continue
            break
        leaves+=bool(child_counts[parent])
        child_counts[parent]+=1;child_counts.append(0)
        d=depth[parent];tokens.append(int(ids[d,k]));parents.append(parent)
        depth.append(d+1);choice.append(k);value.append(-neg)
    return tokens,parents,depth

def segments(parents):
    paths=[[0]];group=[0];first=set()
    for node,p in enumerate(parents[1:],1):
        if p in first:
            group.append(len(paths));paths.append([node])
        else:
            first.add(p);group.append(group[p]);paths[group[p]].append(node)
    return paths

def ancestor(parents,v):
    result=[v]
    while v:
        v=parents[v];result.append(v)
    return result[::-1]

def verify_attention(parents,paths,rng):
    prefix=6
    pos=np.arange(prefix)
    q=rng.normal(size=(8,4));k=rng.normal(size=(prefix+8,4));v=rng.normal(size=(prefix+8,4))
    check=0
    for seg in paths:
        ancestors=ancestor(parents,seg[-1])
        pages=np.r_[pos,prefix+np.array(ancestors)]
        for window in (None,3):
            group_pages=pages if window is None else pages[-(window+len(seg)):]
            for local,node in enumerate(seg):
                # FA3 bottom-right alignment: node i cannot see the last qlen-1-i keys.
                end=len(group_pages)-len(seg)+local+1
                seen=group_pages[:end] if window is None else group_pages[max(0,end-window-1):end]
                independent=np.r_[pos,prefix+np.array(ancestor(parents,node))]
                if window is not None:independent=independent[-(window+1):]
                assert np.array_equal(seen,independent)
                def score(seq):
                    dots=q[node] @ k[seq].T;p=np.exp(dots-dots.max());return (p/p.sum()) @ v[seq]
                assert np.array_equal(score(seen),score(independent))
                check+=1
    return check

def calls(counts):
    i=0;n=0
    while i<len(counts):
        i+=counts[i];n+=1
    return n

def main():
    checked=0;rows=[];rng=np.random.default_rng(11341909)
    for path in sorted((K/"public_pair").glob("trace_*.npz")):
        num=int(path.stem.split("_")[-1]);d=np.load(path);seq=d["tokens"];anc=d["anchors"]
        score=d["scores"].astype(np.float32)
        shift=score-np.max(score,-1,keepdims=True);logp=shift-np.log(np.exp(shift).sum(-1,keepdims=True))
        arms={}
        for cap in (1,2,3,4,7):
            counts=[];leaf=[];active=[];ql=[];topologies=collections.Counter()
            for j,a in enumerate(anc):
                tokens,parents,depth=proposal(d["ids"][j],logp[j],seq[a],cap)
                chains=segments(parents);leaf.append(len(chains)); ql.append([len(s) for s in chains])
                assert len(chains)<=cap and sorted(x for group in chains for x in group)==list(range(8))
                if j%8==0:checked+=verify_attention(parents,chains,rng)
                node=0;gold=seq[a+1:a+8];n=1
                for t in gold:
                    hits=[k for k in range(1,8) if parents[k]==node and tokens[k]==t]
                    if not hits:break
                    assert len(hits)==1;node=hits[0];n+=1
                counts.append(n);topologies[tuple(parents)]+=1
            i=0
            while i<len(anc):
                active.append(leaf[i]);i+=counts[i]
            arms[cap]=dict(calls=calls(counts),mean_leaf_at_visited=float(np.mean(active)),
                            leaf_hist_at_visited=dict(collections.Counter(active)),
                            max_groups=max(leaf),max_query=max(max(x) for x in ql))
        rows.append(dict(id=num,arms=arms))
    scores={}
    for ids in [list(range(0,12,2)),list(range(1,12,2))]:
        sample=[r["arms"] for r in rows if r["id"] in ids]
        name="heldout" if ids[0]==1 else "discovery";scores[name]={}
        for cap in (1,2,3,4,7):
            baseline=np.array([r[1]["calls"] for r in sample]);cand=np.array([r[cap]["calls"] for r in sample])
            picks=np.random.default_rng(1193).integers(0,len(sample),(20000,len(sample)))
            ci=np.percentile(baseline[picks].sum(1)/cand[picks].sum(1),[2.5,97.5])
            scores[name][cap]=dict(chain_calls=int(baseline.sum()),tree_calls=int(cand.sum()),
                call_ratio=float(baseline.sum()/cand.sum()),prompt_bootstrap_interval=ci.tolist(),
                groups_per_visited_step=sum(r[cap]["mean_leaf_at_visited"]*r[cap]["calls"] for r in sample)/cand.sum())
    out=dict(scope="Genuine 12 Qwen3-1.7B+DFlash2 continuations; even/odd fixed request split. Same eight verified nodes per round, no unary coverage, no new target forward. Calls ratio is maximum allowed relative step time. Groups correspond to actual native causal attention sequences with exact bottom-right/window keys. Scores model stored in BF16 then converted to FP32.",oracle_rows=checked,summary=scores,requests=rows)
    (H/"fixed_tree/segments_replay.json").write_text(json.dumps(out,indent=2));print(json.dumps(scores))
if __name__=="__main__":main()

