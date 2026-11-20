"""Upper bound for a fixed-budget speculative tree given mean chain acceptance.

Allows a different tree and target distribution for each example. Assumes the draft
chain follows a most likely target continuation conditional on each spine prefix.
No calibration assumption is made on off-spine transitions. This assumption is
strong and the results apply as a diagnostic rather than to arbitrary Muse logits.
"""
from pathlib import Path
import itertools
import json
import numpy as np
from scipy.optimize import linprog


def tree_bound(mean, nodes=7, width=7, calibrated=True):
    # Group by the retained greedy spine length and widest off-spine fork.
    groups = [(length, fork) for length in range(min(width,nodes)+1)
              for fork in range(1, min(length+1,width)+1)]
    stride = width+2  # group mass, unnormalized survival probabilities, fork mass
    size = stride*len(groups)
    inequalities, targets = [], []
    exact = np.zeros((2,size))
    reward = np.zeros(size)
    for g,(length,fork) in enumerate(groups):
        offset = stride*g
        mass = offset
        best = offset+width+1
        exact[0,mass] = 1
        exact[1,offset+1:offset+width+1] = 1
        reward[offset+1:offset+length+1] = 1
        reward[best] = nodes-length
        def constraint(terms):
            row = np.zeros(size)
            for i,v in terms: row[i] += v
            inequalities.append(row)
            targets.append(0.0)
        for k in range(1,width+1):
            constraint([(offset+k,1),(offset+k-1,-1)])
        constraint([(best,1),(offset+fork,1),(offset+fork-1,-1)])
        if calibrated: constraint([(best,1),(offset+fork,-1)])
    result = linprog(-reward,A_ub=inequalities,b_ub=targets,A_eq=exact,b_eq=[1,mean-1],
                     bounds=(0,None),method="highs")
    if not result.success: raise RuntimeError(result.message)
    return min(width+1,1-result.fun)


def expected(group):
    return sum(np.cumprod(group))+1


def exact_small_test():
    # For a 3-token binary proposal, enumerate all prefix-closed tree choices.
    # Each check compares the LP against the exact best of all valid trees.
    rng=np.random.default_rng(4419)
    checked=0
    for _ in range(20):
        prob={():1.}
        for depth in range(3):
            for node in list(prob):
                if len(node)!=depth: continue
                p=rng.uniform(.5,1.) if all(i==0 for i in node) else rng.random()
                prob[node+(0,)]=prob[node]*p
                prob[node+(1,)]=prob[node]*(1-p)
        keys=tuple(k for k in prob if k)
        chain=1+sum(prob[(0,)*i] for i in range(1,4))
        opt=max(1+sum(prob[n] for n in combo) for combo in itertools.combinations(keys,3)
                if all(n[:-1] in combo or len(n)==1 for n in combo))
        assert tree_bound(chain,3,3)+1e-9>=opt
        checked+=1
    return checked


def main():
    here=Path(__file__).resolve().parent
    result=json.loads((here.parent/"release_2026-11-18/scored_submission_status.json").read_text())["result"]
    score=result["frontier"]["score"]
    rows=[]
    for rung in result["rungs"]:
        mean=rung["acceptance"]
        for nodes in (7,11,15,31):
            best=tree_bound(mean,nodes)
            rows.append(dict(concurrency=rung["concurrency"],mean=mean,proposal_nodes=nodes,
                fixed_work_calibrated_oracle_bound=best,
                optimistic_decode_ratio=best/mean,
                required_tokens_for_1p8_if_decode_only=mean*1.8/score))
    output=dict(parent=score,exhaustive_oracle_tests=exact_small_test(),rows=rows)
    print(json.dumps(output,indent=2))
    (here/"bounds.json").write_text(json.dumps(output,indent=2))
if __name__=="__main__":
    main()
