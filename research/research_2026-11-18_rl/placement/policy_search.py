"""Choose migration admission on one model, validate the selected fixed policy on a distinct model."""
import json
from pathlib import Path
import numpy as np
from placement import NATIVE,select,route_counts,peak,work
from real_traces import LAYERS,STARTS

HERE=Path(__file__).resolve().parent
THRESHOLDS=(1.,1.03,1.08,1.15)
SWAPS=(1,4,8)


def evaluate(name):
    rows=np.load(HERE/(name+"_routes.npz"))["arr"]
    scores={}
    for m in (128,768):
        for budget in SWAPS:
            evals={t:[] for t in THRESHOLDS}
            ideal=[]
            for region in range(len(STARTS)):
                for i,layer in enumerate(LAYERS):
                    ids=rows[region,:,i]
                    counts=route_counts(ids[:1536])
                    proposed,actions=select(counts,budget)
                    validation=[]
                    for start in range(1536,3072,m):
                        local=route_counts(ids[start:start+m])
                        validation.append(peak(local,NATIVE,16)/peak(local,proposed,16))
                    valid=np.percentile(validation,25)
                    for start in range(3072,len(ids)-m+1,m):
                        counts=route_counts(ids[start:start+m])
                        native=peak(counts,NATIVE,16)
                        moved=peak(counts,proposed,16)
                        ideal.append((native,int(np.ceil(work(counts,16).sum()/8))))
                        for threshold in THRESHOLDS:
                            admit=bool(actions) and valid>threshold
                            evals[threshold].append((native,moved if admit else native,admit))
            for t,records in evals.items():
                original,selected,enabled=np.array(records).T
                scores[f"M{m}_swaps{budget}_gate{t}"]=dict(m=m,swap_budget=budget,threshold=t,
                    aggregate_peak_work_ratio=float(original.sum()/selected.sum()),instances=int(len(original)),fraction_admitted=float(enabled.mean()),
                    fraction_worse=float((selected>original).mean()),ideal_work_balance_ceiling=float(np.array(ideal)[:,0].sum()/np.array(ideal)[:,1].sum()))
    return scores


def main():
    discovery=evaluate("glm53")
    # Choose a single policy jointly across sizes. Never select a new shape subset on confirmation.
    candidates=[(np.mean([np.log(discovery[f"M{m}_swaps{b}_gate{t}"]["aggregate_peak_work_ratio"]) for m in (128,768)]),b,t) for b in SWAPS for t in THRESHOLDS]
    score,b,t=max(candidates)
    confirmation=evaluate("glm52")
    chosen={name:[rows[f"M{m}_swaps{b}_gate{t}"] for m in (128,768)] for name,rows in (("discovery",discovery),("confirmation",confirmation))}
    result=dict(scope="Ratios of summed peak routed-expert tile counts on real external GLM decode traces. Per-token costs and communication are excluded. Expert counts from earlier tokens only.",chosen=chosen,all_discovery=discovery,all_confirmation=confirmation)
    (HERE/"policy_search.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(chosen))

if __name__=="__main__":main()
