"""Select a linear DFlash path maximizing expected committed tokens under its existing transition lattice."""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
TEMPERATURES=(.50,.75,1.,1.50,2.)
DISCOUNTS=(0.,.25,.50,.75,1.)
def policy(scores,temperature,gamma):
    n,d,k,k2=scores.shape
    assert k==k2
    vals=np.zeros((n,k),dtype=np.float32)
    actions=np.empty((n,d,k),dtype=np.int32)
    for j in reversed(range(d)):
        s=scores[:,j].astype(np.float32)
        if gamma==0:
            actions[:,j]=s.argmax(-1)
            continue
        s=(s-s.max(-1,keepdims=True))/temperature
        p=np.exp(s);p/=p.sum(-1,keepdims=True)
        expected=p*(1.+gamma*vals[:,None,:])
        actions[:,j]=expected.argmax(-1)
        vals=expected.max(-1)
    prev=np.zeros(n,dtype=np.int32);chain=[]
    for j in range(d):
        prev=actions[np.arange(n),j,prev]
        chain.append(prev)
    return np.stack(chain,1)

def counts(data,choice):
    rows=np.arange(len(choice))[:,None]; steps=np.arange(choice.shape[-1])[None,:]
    predicted=data["ids"][rows,steps,choice]
    target=data["tokens"][data["anchors"][:,None]+steps+1]
    return (predicted==target).cumprod(1).sum(1)

def calls(correct):
    i=n=0
    while i<len(correct):
        i+=1+int(correct[i]);n+=1
    return n

def main():
    paths=sorted(HERE.glob("trace_*.npz"))
    assert len(paths)==12
    choices=[(1.,0.)]+[(t,g) for t in TEMPERATURES for g in DISCOUNTS if g>0]
    rows=[]
    for pp in paths:
        data=dict(np.load(pp))
        row={}
        ref=policy(data["scores"],1.,0.)
        for t,g in choices:
            chosen=policy(data["scores"],t,g)
            correct=counts(data,chosen)
            key=f"t{t:.2f}_g{g:.2f}"
            row[key]=dict(calls=calls(correct),mean_commits=float(correct.mean()+1),
                          changed_anchors=int((ref!=chosen).any(1).sum()))
        rows.append(row)
    train=list(range(0,len(rows),2));held=list(range(1,len(rows),2))
    base="t1.00_g0.00"
    def ratio(ids,arm):
        return sum(rows[i][base]["calls"] for i in ids)/sum(rows[i][arm]["calls"] for i in ids)
    # Fixed default wins every tie. No held-out request participates in selection.
    name=max(rows[0],key=lambda k:ratio(train,k))
    fixed="t1.00_g1.00"
    rng=np.random.default_rng(7841)
    studies={}
    for key in {fixed,name}:
        ci=np.quantile([ratio(rng.choice(held,len(held),True),key) for _ in range(2000)],[.025,.975])
        studies[key]=dict(discovery=ratio(train,key),holdout=ratio(held,key),holdout_interval=ci.tolist(),
                          per_request=[ratio([i],key) for i in held],parent_calls=sum(rows[i][base]["calls"] for i in held),
                          candidate_calls=sum(rows[i][key]["calls"] for i in held))
    result=dict(scope="Real public Qwen3/DFlash2 traces. Target verification stays a fixed eight-row linear chain. The unit is calls at identical output length on request-disjoint trajectories, before measuring planner or service cost.",
                selected=name,default_bellman=fixed,train=train,holdout=held,study=studies,per_request=rows)
    (HERE/"bellman.json").write_text(json.dumps(result,indent=2))
    print(name,studies)
if __name__=="__main__":main()
