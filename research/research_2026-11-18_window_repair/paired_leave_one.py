"""Per-prompt cross-fitting of the frozen conditional model and true serial greedy path."""
import json
import numpy as np
from paired_confidence import PAIR,HERE,read_prompt,fit,prediction,replay,RATIOS
from paired_lagged import metrics,visit
from paired_confidence import sigmoid

def main():
    files=sorted(PAIR.glob("trace_[0-9][0-9].npz"))
    items={int(f.stem[-2:]):read_prompt(f) for f in files}
    assert len(items)>=5
    all_cases=[]
    for idx,test in items.items():
        train=[p for i,p in items.items() if i!=idx]
        prior=fit(train,2);post=fit(train,4)
        x=np.stack([ex[0] for ex in test["examples"]]);y=np.array([ex[1] for ex in test["examples"]])
        proper={name:metrics(y,prediction(x,name,prior,post)) for name in ("position_prior","raw_score","trained_score")}
        decoded={}
        for ratio in RATIOS:
            # Match evaluated per-user shape by independent serial discovery replays.
            def train_time(w):
                return sum(replay([p],w,ratio,prior,post)["normalized_time"] for p in train)
            fixed=min((0,1,3,7),key=train_time)
            cases={str(w):replay([test],w,ratio,prior,post) for w in (0,1,3,7,"trained_score")}
            decoded[str(ratio)]=dict(discovery_fixed=fixed,
                                     post=cases["trained_score"]["normalized_time"],
                                     selected=cases[str(fixed)]["normalized_time"],
                                     full=cases["7"]["normalized_time"])
        first=3
        seen=[visit(p,first) for p in train]; held=visit(test,first)
        q0=fit(seen,2);q1=fit(seen,5)
        xx=np.stack([ex[0] for ex in held["examples"]]);yy=np.asarray([ex[1] for ex in held["examples"]])
        pre=dict(lagged=metrics(yy,sigmoid(xx@q1)),position=metrics(yy,sigmoid(xx[:,:2]@q0)))
        all_cases.append(dict(prompt=idx,heldout_reached=len(y),proper=proper,decoded=decoded,lagged_prior_cap3=pre))
    summary={}
    for ratio in map(str,RATIOS):
        x=[v["decoded"][ratio] for v in all_cases]
        rel=[v["selected"]/v["post"] for v in x]
        summary[ratio]=dict(equal_tokens_pooled_vs_calibrated_fixed=sum(v["selected"] for v in x)/sum(v["post"] for v in x),
                            equal_tokens_pooled_vs_full=sum(v["full"] for v in x)/sum(v["post"] for v in x),
                            smallest=min(rel),largest=max(rel),improved_prompts=sum(n>1 for n in rel))
    doc=dict(scope="Leave exactly one whole prompt out; unchanged features, penalty and graph rungs declared before outputs. Actual Qwen3-1.7B greedy response, per-user serial paths, assumed timing curves; post-hoc cross-validation supplement to frozen modulo split.",
             prompts=len(all_cases),proper=all_cases,time_sensitivity=summary)
    (HERE/f"decode_selector/qwen_lopo_{len(items)}.json").write_text(json.dumps(doc,indent=2))
    print(len(items),{k:round(v["equal_tokens_pooled_vs_calibrated_fixed"],3) for k,v in summary.items()})
if __name__=="__main__":main()
