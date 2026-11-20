"""Offline fixed-capacity expert-swap sensitivity; no model or serving changes."""
import json
import math
import random
from pathlib import Path
from statistics import median

RANKS, PER_RANK, TOP_K, TILE = 8, 32, 8, 16
EXPERTS = RANKS * PER_RANK

def traffic(seed, tokens, batches, hot):
    rng = random.Random(seed)
    histograms = []
    for _ in range(batches):
        counts = [0] * EXPERTS
        for _ in range(tokens):
            selected = set()
            while len(selected) < TOP_K:
                expert = rng.choice(hot) if hot and rng.random() < 0.55 else rng.randrange(EXPERTS)
                selected.add(expert)
            for expert in selected:
                counts[expert] += 1
        assert sum(counts) == tokens * TOP_K
        histograms.append(counts)
    return histograms

def tiles(counts):
    return [(v + TILE - 1)//TILE for v in counts]

def rank_work(cost, owners):
    work = [0]*RANKS
    for expert,rank in enumerate(owners):
        work[rank] += cost[expert]
    return work

def fit(history):
    cost = list(map(sum, zip(*(tiles(c) for c in history))))
    owners = [e//PER_RANK for e in range(EXPERTS)]
    moved = set()
    for _ in range(8):
        work = rank_work(cost, owners)
        best = tuple(sorted(work,reverse=True))
        choice = None
        for a in range(EXPERTS):
            if a in moved:
                continue
            for b in range(a+1,EXPERTS):
                i,j = owners[a],owners[b]
                if i==j or b in moved:
                    continue
                delta = cost[a] - cost[b]
                if delta*(work[i]-work[j]) <= 0:
                    continue
                after = work.copy()
                after[i] -= delta
                after[j] += delta
                obj = tuple(sorted(after,reverse=True))
                if obj < best:
                    best,choice = obj,(a,b)
        if choice is None:
            break
        a,b=choice
        owners[a],owners[b]=owners[b],owners[a]
        moved.update(choice)
    assert all(owners.count(r)==PER_RANK for r in range(RANKS))
    return owners,len(moved)//2

def main():
    base = [e//PER_RANK for e in range(EXPERTS)]
    clustered = list(range(16))
    spread = [32*r+i for r in range(RANKS) for i in range(2)]
    cases = [
        ("uniform",[],[]),
        ("persistent_cluster",clustered,clustered),
        ("spread_hot",spread,spread),
        ("cluster_moves",clustered,[x+64 for x in clustered]),
        ("cluster_dissipates",clustered,[]),
    ]
    out=[]
    for tokens in (8,128,753):
        for label,train_hot,test_hot in cases:
            train=traffic(27,tokens,20,train_hot)
            test=traffic(81,tokens,40,test_hot)
            owners,swaps=fit(train)
            old=new=ideal=0
            samples=[]
            for c in test:
                costs=tiles(c)
                before=max(rank_work(costs,base))
                after=max(rank_work(costs,owners))
                floor=max(max(costs),math.ceil(sum(costs)/RANKS))
                old+=before
                new+=after
                ideal+=floor
                samples.append(before/after)
            out.append(dict(tokens=tokens,pattern=label,swaps=swaps,baseline_to_swap=old/new,
                            optimistic_upper_bound=old/ideal,worse_batches=sum(x<1 for x in samples),
                            batches=len(samples)))
    result=dict(scope="Synthetic unique global top-8. One tile per 16 routed rows per expert. Maximum rank work per batch, summed across held-out batches. Excludes latency, weights, communication, and any measured judge routing distribution.",cases=out)
    Path(__file__).with_name("placement_bounds.json").write_text(json.dumps(result,indent=2))
    for entry in out:
        print(entry["tokens"],entry["pattern"],round(entry["baseline_to_swap"],3),round(entry["optimistic_upper_bound"],3))
if __name__=="__main__":
    main()
