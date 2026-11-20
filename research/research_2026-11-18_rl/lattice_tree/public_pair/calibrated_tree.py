"""Fit a rank- and depth-dependent proposal model on even requests and test on unopened odd requests."""
import heapq
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
TEMPS=np.array([.40,.55,.70,.85,1.,1.30,1.65,2.20,3.0])
NODES=(8,12,16)

def records(data):
    ids=data["ids"]; scores=data["scores"].astype(np.float32)
    positions=data["anchors"]; tokens=data["tokens"]
    mass=data["log_unary_mass"]
    gold=tokens[positions[:,None]+np.arange(1,8)[None]]
    for i in range(len(ids)):
        previous=0;fork=False
        for j in range(7):
            row=scores[i,j,previous]
            hit=np.flatnonzero(ids[i,j]==gold[i,j])
            selected=int(hit[0]) if len(hit) else -1
            yield j,int(fork),row,selected,float(mass[i,j])
            if selected<0:break
            fork=fork or selected!=int(row.argmax())
            previous=selected

def loss(row,selected):
    r=(row-row.max())[None,:]/TEMPS[:,None]
    return np.log(np.exp(r).sum(1))-r[:,selected]

def fit(inputs):
    # A weak unit-temperature prior and neighboring population shrinkage constrain sparse depth/fork bins.
    n=np.zeros((7,2),dtype=np.int32)
    hits=np.zeros_like(n)
    nll=np.zeros((7,2,len(TEMPS)))
    means=np.zeros_like(n,dtype=np.float64)
    for data in inputs:
        for j,f,row,k,logmass in records(data):
            n[j,f]+=1
            means[j,f]+=min(1.,np.exp(logmass))
            if k>=0:
                hits[j,f]+=1
                nll[j,f]+=loss(row,k)
    result=[]
    all_loss=nll.sum((0,1))/max(1,hits.sum())
    for j in range(7):
        per=[]
        for f in range(2):
            nearest=nll[j].sum(0)/max(1,hits[j].sum())
            blended=nll[j,f]+12*nearest+10*all_loss+4*np.log(TEMPS)**2
            temperature=float(TEMPS[blended.argmin()])
            base_rate=float((hits[j].sum()+2)/(n[j].sum()+4))
            coverage=float((hits[j,f]+12*base_rate)/(n[j,f]+12))
            observed=float((means[j,f]+12*(means[j].sum()+1)/(n[j].sum()+2))/(n[j,f]+12))
            per.append(dict(temperature=temperature,coverage=coverage,mean_unary_mass=observed,
                            reached=int(n[j,f]),in_top16=int(hits[j,f])))
        result.append(per)
    return result

def calls(count):
    i=0;n=0
    while i<len(count):
        i+=int(count[i])+1;n+=1
    return n

def solve(ids,score,mass,gold,model,adjust):
    heap=[]
    def expand(depth,prev,forked,value,path):
        par=model[depth][int(forked)]
        r=score[depth,prev].astype(np.float64)
        r=(r-r.max())/par["temperature"]
        logprob=r-np.log(np.exp(r).sum())
        coverage=par["coverage"]
        if adjust:
            coverage*=min(1.,np.exp(mass[depth]))/max(.01,par["mean_unary_mass"])
        logprob+=np.log(np.clip(coverage,.000001,1.))
        greedy=int(r.argmax())
        for k in range(16):
            child=path+(int(ids[depth,k]),)
            heapq.heappush(heap,(-value-logprob[k],depth+1,child,k,forked or k!=greedy))
    expand(0,0,False,0.,())
    matched=0;counts={}
    for node in range(1,max(NODES)):
        minus,d,path,child,f=heapq.heappop(heap)
        if path==gold[:d]:
            matched=max(matched,d)
        if node+1 in NODES:
            counts[node+1]=matched
        if d<7:expand(d,child,f,-minus,path)
    return counts

def evaluate(data,model,adjust):
    chain=[];counts={n:[] for n in NODES}
    for i,pos in enumerate(data["anchors"]):
        ids=data["ids"][i];scores=data["scores"][i].astype(np.float32)
        gold=tuple(map(int,data["tokens"][pos+1:pos+8]))
        m=0;prev=0
        for j in range(7):
            prev=int(scores[j,prev].argmax())
            if ids[j,prev]!=gold[j]:break
            m+=1
        chain.append(m)
        matched=solve(ids,scores,data["log_unary_mass"][i],gold,model,adjust)
        for n in NODES:counts[n].append(matched[n])
    return {"chain":calls(chain),**{str(n):calls(counts[n]) for n in NODES}}

def main():
    files=sorted(HERE.glob("trace_*.npz"))
    assert len(files)==12
    # Materialize zip members once, outside scoring loops.
    data=[dict(np.load(p)) for p in files]
    tr=[i for i in range(12) if i%2==0]
    te=[i for i in range(12) if i%2==1]
    model=fit([data[i] for i in tr])
    rows=[{str(a):evaluate(d,model,a) for a in (0,1)} for d in data]
    def ratio(indices,n,adjust):
        group=[rows[i][str(adjust)] for i in indices]
        return sum(r["chain"] for r in group)/sum(r[str(n)] for r in group)
    selected={}
    for n in NODES:
        adj=max((0,1),key=lambda a:ratio(tr,n,a))
        rng=np.random.default_rng(26719)
        bounds=np.quantile([ratio(rng.choice(te,len(te),replace=True),n,adj) for _ in range(2000)],[.025,.975])
        selected[n]=dict(adjust_mass=bool(adj),train_call_ratio=ratio(tr,n,adj),held_call_ratio=ratio(te,n,adj),
                         held_interval=bounds.tolist(),per_request=[ratio([i],n,adj) for i in te])
    result=dict(scope="Matched Qwen3 target/draft replay. A ratio counts verifier calls at equal token counts. Each request stays intact in the holdout.",
                fitted=model,selected=selected,rows=rows,train=tr,held=te)
    (HERE/"calibrated_tree.json").write_text(json.dumps(result,indent=2))
    print(selected)
if __name__=="__main__": main()
