"""Ask whether the previous finished block contains useful BEFORE-draft acceptance evidence."""
from pathlib import Path
import argparse
import json
import numpy as np
from paired_confidence import HERE,PAIR,read_prompt,fit,sigmoid

def features(previous):
    # Before first draft these are population-centred constants. All subsequent
    # features use only the previous call's observed, possibly truncated result.
    vals = [0.,0.,0.] if previous is None else previous
    return np.array([[1., (j-3)/3., *vals] for j in range(7)])

def visit(p,width):
    pos=p["start"]; previous=None; examples=[]; indices=[]
    while pos in p["rounds"]:
        truth,source=p["rounds"][pos]
        x=features(previous)
        for j in range(min(7,truth+1)):
            examples.append((x[j],int(j<truth)))
            indices.append(j)
        seen=min(width,truth)
        # This score is from the already finished previous draft's first slot.
        # Current-width post scores cannot be consumed until the next call.
        previous=[source[0,2], (seen-2)/3.,float(seen<width)]
        pos+=1+seen
    return dict(examples=examples,slots=indices)

def metrics(y,p):
    p=np.clip(p,1e-7,1-1e-7)
    return dict(brier=float(np.mean((p-y)**2)),nll=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--folder",type=Path,default=PAIR);ap.add_argument("--out",type=Path,default=HERE/"decode_selector/qwen_lagged.json");a=ap.parse_args()
    prompts={int(f.stem.split("_")[-1]): read_prompt(f) for f in a.folder.glob("trace_[0-9][0-9].npz")}
    assert sum(i%3==2 for i in prompts)>=2
    result={}
    for width in (1,3,7):
        train=[visit(p,width) for i,p in prompts.items() if i%3!=2]
        test=[visit(p,width) for i,p in prompts.items() if i%3==2]
        prior=fit(train,2); lagged=fit(train,5)
        x=np.stack([j[0] for seq in test for j in seq["examples"]]);y=np.array([j[1] for seq in test for j in seq["examples"]])
        result[width]=dict(reached_labels=len(y),prior=metrics(y,sigmoid(x[:,:2]@prior)),lagged=metrics(y,sigmoid(x@lagged)),beta=lagged.tolist())
    doc=dict(scope="One-step-lagged acceptance predictor. The previous call's observed (censored) acceptance and its first selector score are available before the next draft. All drafts in the trace physically use width 8; varying cap only changes which greedy verifier rows are observed. Actual changed-size DFlash bidirectional forward outputs are unknown.",
        test_ids=[i for i in prompts if i%3==2],fit_ids=[i for i in prompts if i%3!=2],previous_verifier_draft_caps=result)
    a.out.write_text(json.dumps(doc,indent=2))
    print([(w,round(r["prior"]["nll"],3),round(r["lagged"]["nll"],3)) for w,r in result.items()])
if __name__=="__main__":main()
