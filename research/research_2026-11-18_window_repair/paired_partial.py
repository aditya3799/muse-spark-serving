"""Report the first fixed holdout promptly, without promoting or tuning further."""
import json
import numpy as np
from paired_confidence import PAIR,HERE,read_prompt,fit,prediction,replay,RATIOS
from paired_lagged import visit,metrics

def main():
    files=[PAIR/f"trace_{i:02d}.npz" for i in (0,1,2)]
    train=[read_prompt(f) for f in files[:2]]
    held=[read_prompt(files[2])]
    prior=fit(train,2);post=fit(train,4)
    x=np.stack([z[0] for z in held[0]["examples"]]);y=np.array([z[1] for z in held[0]["examples"]])
    proper={k:metrics(y,prediction(x,k,prior,post)) for k in ("position_prior","raw_score","trained_score")}
    sensitivity={}
    for ratio in RATIOS:
        gold=min((0,1,3,7),key=lambda w:replay(train,w,ratio,prior,post)["normalized_time"])
        runs={str(p):replay(held,p,ratio,prior,post) for p in (0,1,3,7,"position_prior","raw_score","trained_score")}
        sensitivity[ratio]=dict(discovery_fixed_slots=gold,raw=runs,post_vs_full8=runs["7"]["normalized_time"]/runs["trained_score"]["normalized_time"],
                                post_vs_tunedfixed=runs[str(gold)]["normalized_time"]/runs["trained_score"]["normalized_time"])
    before={}
    for width in (1,3,7):
        seen=[visit(p,width) for p in train];actual=visit(held[0],width)
        coef=fit(seen,5);base=fit(seen,2)
        x0=np.stack([item[0] for item in actual["examples"]]);y0=np.array([item[1] for item in actual["examples"]])
        from paired_confidence import sigmoid
        before[width]=dict(labels=len(y0),lagged=metrics(y0,sigmoid(x0@coef)),position_prior=metrics(y0,sigmoid(x0[:,:2]@base)))
    out=dict(scope="Preliminary first whole held-out prompt from the frozen modulo-three split. Qwen3 1.7B paired model, greedy generation. All rate estimates are synthetic graph cost sensitivities. The independent study requires at least two held-out prompts.",
             discovery=[0,1],holdout=[2],labels=len(y),heldout_anchors=len(held[0]["rounds"]),proper=proper,before=before,modeled=sensitivity)
    (HERE/"decode_selector/qwen_partial_prompt2.json").write_text(json.dumps(out,indent=2))
    print(out["labels"],proper,{r:round(o["post_vs_tunedfixed"],3) for r,o in sensitivity.items()})
if __name__=="__main__":main()
