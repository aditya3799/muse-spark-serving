"""Evaluate only request-disjoint tree policies on real paired greedy target and DFlash2 traces."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parent))
from selector_tree import build_tree

def accepted_chain(ids,scores,gold):
    count=len(ids)
    prev=np.zeros(count,dtype=int); correct=np.zeros(count,dtype=int)
    for j in range(7):
        prev=scores[np.arange(count),j,prev].argmax(axis=-1)
        picked=ids[np.arange(count),j,prev]
        correct+=(correct==j)&(picked==gold[:,j])
    return correct
def accepted_tree(ids,scores,mass,anchors,tokens,gold,n,temp,coverage,fork=1.):
    vals=torch.from_numpy(scores).float()/temp
    covered=(torch.from_numpy(mass).float().exp()[:,:,None].expand(-1,-1,16) if coverage else None)
    t,p,_,_=build_tree(torch.from_numpy(ids).long(),vals,torch.from_numpy(tokens[anchors]).long(),n,covered,after_fork=fork)
    t,p=t.numpy(),p.numpy()
    counts=np.zeros(len(ids),dtype=int)
    for i in range(len(ids)):
        root=0
        for token in gold[i]:
            child=np.flatnonzero((p[i]==root)&(t[i]==token))
            child=child[child>0]
            if not len(child):break
            assert len(child)==1
            root=child[0];counts[i]+=1
    return counts
def replay_count(correct):
    position=0;calls=0;total=len(correct)
    while position<total:
        position+=min(int(correct[position])+1,total-position);calls+=1
    return calls
def per_prompt(path):
    d=np.load(path)
    ids=d["ids"]; scores=d["scores"].astype(np.float32); tokens=d["tokens"]; anchor=d["anchors"]
    assert np.array_equal(np.diff(anchor),np.ones(len(anchor)-1))
    gold=tokens[anchor[:,None]+np.arange(1,8)[None]]
    observed=accepted_chain(ids,scores,gold)
    rows={"chain8":dict(calls=replay_count(observed),mean_c=int(observed.sum())/len(anchor)+1)}
    # Oracle retains the observed target ID at each layer whenever it exists among the proposed top 16.
    oracle=np.cumprod((ids==gold[:,:,None]).any(-1),axis=1).sum(1)
    rows["oracle"]=dict(calls=replay_count(oracle),mean_c=float(oracle.mean()+1))
    for n in (8,12,16):
        for temp in (.65,1.,1.5):
            for cov in (False,True):
                for fork in (1.,.5):
                    c=accepted_tree(ids,scores,d["log_unary_mass"],anchor,tokens,gold,n,temp,cov,fork)
                    name=f"n{n}_t{temp:.2f}_m{int(cov)}_f{fork:.2f}"
                    rows[name]=dict(calls=replay_count(c),mean_c=float(c.mean()+1))
    return dict(prompt=int(path.stem.split("_")[-1]),anchors=len(anchor),arms=rows)
def main():
    torch.set_num_threads(2)
    rows=[per_prompt(p) for p in sorted(here.glob("trace_*.npz"))]
    if len(rows)<2:return
    def gain(data,key):
        return sum(r["arms"]["chain8"]["calls"] for r in data)/sum(r["arms"][key]["calls"] for r in data)
    train=[r for r in rows if r["prompt"]%2==0];held=[r for r in rows if r["prompt"]%2==1]
    if not train or not held:return
    choice={}
    for n in (8,12,16):
        label=max([k for k in train[0]["arms"] if k.startswith(f"n{n}_")],key=lambda k:gain(train,k))
        rng=np.random.default_rng(8617)
        boot=[gain([held[i] for i in rng.integers(len(held),size=len(held))],label) for _ in range(2000)]
        choice[n]=dict(policy=label,discovery_call_ratio=gain(train,label),holdout_call_ratio=gain(held,label),
                       holdout_call_ratio_interval=np.quantile(boot,[.025,.975]).tolist(),
                       holdout_prompt_ids=[r["prompt"] for r in held],holdout_ratios=[gain([r],label) for r in held])
    result=dict(scope="Real serial greedy tokens using public Qwen3-1.7B, its trained DFlash2 and all top-16 predecessor edges. Ratios compare verification calls at identical output lengths. Mean counts at all anchors and per-policy scheduling are reported separately. A tree adds causal mask and planning work; the call ratio is its break-even step-time multiplier, not an end-to-end acceleration.",
                trace_count=len(rows),holdout_upper_bound_ideal_path=gain(held,"oracle"),selected=choice,requests=rows)
    (here/"policy_replay.json").write_text(json.dumps(result,indent=2))
    print({n:(r["policy"],r["holdout_call_ratio"]) for n,r in choice.items()})
if __name__=="__main__":main()