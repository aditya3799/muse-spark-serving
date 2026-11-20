"""Prompt-held-out conditional acceptance and consecutive greedy replay on the real public Qwen pair."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAIR = HERE.parent / "research_2026-11-18_rl/lattice_tree/public_pair"
RATIOS = (1.25, 1.5, 2.0, 3.0, 5.0)


def read_prompt(path):
    with np.load(path) as data:
        ids = data["ids"]
        scores = data["scores"].astype(np.float64)
        anchors = data["anchors"]
        tokens = data["tokens"]
        logmass = data["log_unary_mass"]
        assert scores.shape == (*ids.shape, 16)
        assert np.array_equal(anchors, np.arange(int(data["start"]), len(tokens) - 7))
        result = {}
        examples = []
        raw_by_position = [[] for _ in range(7)]
        for row, anchor in enumerate(anchors):
            prev = 0
            accepts = 0
            alive = True
            features = []
            for slot in range(7):
                v = scores[row, slot, prev]
                ix = np.argmax(v)
                probs = np.exp(v - v.max())
                p = probs[ix] / probs.sum()
                # Fixed, nonnegative-scale features. The bias and slot term form the prior.
                logit_p = np.log(np.clip(p,1.e-7,1.-1.e-7) / np.clip(1. - p,1.e-7,1.))
                x = [1., (slot - 3) / 3., np.clip(logit_p, -6, 6) / 3., np.clip(logmass[row, slot], -6, 0) / 3.]
                features.append(x)
                match = bool(ids[row,slot,ix] == tokens[anchor + 1 + slot])
                if alive:
                    examples.append((x, int(match)))
                    raw_by_position[slot].append((p,match))
                    if match:
                        accepts += 1
                    else:
                        alive = False
                prev = ix
            result[int(anchor)] = (accepts, np.array(features))
        return dict(start=int(data["start"]),end=len(tokens)-1,rounds=result,
                    examples=examples,by_position=raw_by_position)


def sigmoid(v):
    return 1. / (1. + np.exp(-np.clip(v,-30.,30.)))


def fit(prompts, width):
    x = np.stack([j[0][:width] for p in prompts for j in p["examples"]])
    y = np.array([j[1] for p in prompts for j in p["examples"]])
    beta = np.zeros(width)
    penalty = np.diag([.1] + [3.] * (width-1))
    for _ in range(32):
        p = sigmoid(x @ beta)
        curvature = x.T @ (x * (p * (1. - p))[:, None]) + penalty
        step = np.linalg.solve(curvature, x.T @ (p - y) + penalty @ beta)
        beta -= step
        if np.max(np.abs(step)) < 1.e-7:
            break
    return beta


def prediction(features, policy, prior, post):
    if policy == "position_prior":
        return sigmoid(features[:, :2] @ prior)
    if policy == "raw_score":
        return sigmoid(features[:, 2] * 3.)
    return sigmoid(features @ post)


def round_slots(probs, ratio):
    # Include the one mandatory anchor per request; target graph rounds to catalog.
    batch = len(probs)
    survival = np.cumprod(probs, axis=1)
    order = np.argsort(-survival.ravel(), kind="stable")
    emitted = batch + np.r_[0.,np.cumsum(survival.ravel()[order])]
    i = np.arange(batch*7+1)
    rungs = np.array([0, batch, 3*batch, 7*batch])
    billed = rungs[np.searchsorted(rungs, i)]
    elapsed = 1. + (ratio-1.) * billed / (7.*batch)
    take = int(np.argmax(emitted / elapsed))
    return np.bincount(order[:take]//7, minlength=batch)


def replay(prompts, policy, ratio, prior, post):
    pos = np.array([p["start"] for p in prompts],dtype=int)
    steps, rows, delivered, elapsed = 0, 0, 0, 0.
    while True:
        active = [i for i,p in enumerate(prompts) if pos[i] < p["end"]]
        if not active:break
        b = len(active)
        uses = []
        probs = []
        for i in active:
            x = prompts[i]["rounds"].get(int(pos[i]))
            # Last six anchors were not recorded. All policies use identical
            # ordinary one-token tail completions from those states and pay a step.
            uses.append(0 if x is None else x[0])
            probs.append(np.zeros(7) if x is None else prediction(x[1],policy,prior,post))
        lengths = np.array([0 if int(pos[i]) not in prompts[i]["rounds"] else 7 for i in active])
        if isinstance(policy,int):
            width = np.minimum(policy,lengths)
        else:
            width = np.minimum(round_slots(np.array(probs),ratio),lengths)
        billed = int(np.array([0,b,3*b,7*b])[np.searchsorted([0,b,3*b,7*b],sum(width))])
        elapsed += 1. + (ratio-1.) * billed / (7.*b)
        add = np.minimum(1+np.minimum(width,uses),[prompts[i]["end"]-pos[i] for i in active])
        pos[active] += add
        steps += 1
        rows += b + billed
        delivered += int(sum(add))
    return dict(steps=steps,billed_rows=rows,emitted=delivered,normalized_time=elapsed)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--folder",type=Path,default=PAIR)
    ap.add_argument("--out",type=Path,default=HERE/"decode_selector/qwen_paired_confidence.json")
    a=ap.parse_args()
    files=sorted(a.folder.glob("trace_[0-9][0-9].npz"))
    by_id={int(f.stem.split("_")[-1]):read_prompt(f) for f in files}
    # Frozen before any NPZ traces existed. Split only by complete prompt.
    train=[p for i,p in by_id.items() if i%3!=2]
    test=[p for i,p in by_id.items() if i%3==2]
    if len(train)<2 or len(test)<2:raise SystemExit("Need two distinct held-out prompts.")
    prior=fit(train,2);post=fit(train,4)
    features=np.stack([ex[0] for p in test for ex in p["examples"]])
    y=np.array([ex[1] for p in test for ex in p["examples"]])
    summary={}
    for name in ("position_prior","raw_score","trained_score"):
        p=np.clip(prediction(features,name,prior,post),1.e-7,1.-1.e-7)
        summary[name]=dict(brier=float(np.mean((p-y)**2)),nll=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))))
    simulations={}
    for r in RATIOS:
        # Choose a single fixed verification width on discovery prompts only.
        fixed=min((0,1,3,7),key=lambda w:replay(train,w,r,prior,post)["normalized_time"])
        runs={str(policy):replay(test,policy,r,prior,post) for policy in (0,1,3,7,"raw_score","position_prior","trained_score")}
        denom=runs[str(fixed)]["normalized_time"]
        for d in runs.values():
            d["vs_selected_fixed"]=denom/d["normalized_time"]
        assert len({d["emitted"] for d in runs.values()})==1
        simulations[str(r)]=dict(train_selected_fixed_draft_slots=fixed,runs=runs)
    result=dict(scope="Teacher-forced reconstruction of serial greedy emissions on the public Qwen3-1.7B paired DFlash2 surrogate. Relative times assume fixed floor (including draft) and 1/2/4/8-equivalent row billing; no GPU latency here.",
                prompt_split="fixed prompt_id%3==2 held out",fit_ids=[i for i in by_id if i%3!=2],test_ids=[i for i in by_id if i%3==2],
                hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                heldout_anchors=sum(len(p["rounds"]) for p in test),heldout_reached_labels=len(y),
                reached_fraction_positive=float(y.mean()),validation=summary,projection_ridge_beta=post.tolist(),
                time_sensitivity=simulations,sampling_scope="Target is greedy. A stochastic acceptance test requires true target distributions for the realized decoder path.")
    a.out.write_text(json.dumps(result,indent=2))
    print("prompts",len(train),len(test),"labels",len(y),summary)
    print("modeled post / selected fixed",{r:round(v["runs"]["trained_score"]["vs_selected_fixed"],3) for r,v in simulations.items()})


if __name__=="__main__":
    main()
