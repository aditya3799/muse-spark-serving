"""Evaluate a frozen own-history policy on two independent public coding corpora.

Synthetic fallback lengths model sensitivity to DFlash being more accurate on easy
copy spans. They are calibrated to the same corpus-wide baseline and are not model
logits, wall times, or reproduced judge requests.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq
from numba import njit
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parent
OTHER=ROOT.parent/"research_2026-11-18_swarm_speculation"
sys.path.insert(0,str(OTHER))
from copy_replay import per_turn

@njit
def sweep(matches,uniform,ends,groups,n_trace,easy,hard,mixed):
    counts=np.zeros((n_trace,4),np.int64)
    start=0
    for t in range(len(ends)):
        stop=ends[t]
        i=start
        a=groups[t]
        while i<stop:
            eligible=matches[i]!=255
            counts[a,0]+=1
            if mixed and eligible and stop-i>=32:
                take=1+int(matches[i])
                counts[a,1]+=1
                counts[a,2]+=min(stop-i,take)
            else:
                mean=easy if eligible else hard
                take=int(mean)+(1 if uniform[i]<mean-int(mean) else 0)
            counts[a,3]+=min(stop-i,take)
            i+=take
        start=stop
    return counts

def dataset(name):
    if name in ("muse_split","muse_joined"):
        from muse_records import samples
        yield from samples(joined=name=="muse_joined")
    elif name=="openhands":
        for row in pq.read_table(ROOT/"openhands.parquet",columns=["messages"]).to_pylist():
            yield row["messages"]
    else:
        for offset in (1000,5000,12000,20000):
            for row in json.loads((OTHER/f"traces_{offset}.json").read_text(encoding="utf8"))["rows"]:
                yield json.loads(row["row"]["messages"])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("dataset",choices=["openhands","swesmith","muse_split","muse_joined"])
    name=p.parse_args().dataset
    tokenizer_path=ROOT/"muse_tokenizer.json"
    digest=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    assert digest=="ac2f41ef2444dec82fdc90e334f80543f298535005fe78aec7e8311ef86f84d0"
    tok=Tokenizer.from_file(str(tokenizer_path))
    turns,groups=[],[]
    n=0
    for trace,msgs in enumerate(dataset(name)):
        n=trace+1
        for rows in per_turn(msgs,tok):
            if not rows: continue
            turns.append(np.fromiter((m if k>=8 and avail>=31 else 255 for k,m,avail in rows),np.uint8))
            groups.append(trace)
    ends=np.cumsum([len(t) for t in turns])
    matches=np.concatenate(turns)
    groups=np.array(groups)
    rng=np.random.default_rng(20261118)
    uniform=rng.random(len(matches),dtype=np.float32)
    bootstrap=rng.integers(n,size=(2000,n))
    results=[]
    for easy in (2.,4.,6.,8.):
        lo,hi=1.,8.
        for _ in range(16):
            hard=(lo+hi)/2
            base=sweep(matches,uniform,ends,groups,n,easy,hard,False)
            if base[:,3].sum()/base[:,0].sum()<4: lo=hard
            else: hi=hard
        counts=sweep(matches,uniform,ends,groups,n,easy,hard,True)
        assert counts[:,3].sum()==base[:,3].sum()==len(matches)
        b,steps,hits=base[:,0],counts[:,0],counts[:,1]
        predictions={}
        for cost in (1,1.25,1.5,2,3,4):
            trial=steps+(cost-1)*hits
            draws=b[bootstrap].sum(1)/trial[bootstrap].sum(1)
            predictions[str(cost)]=dict(ratio=float(b.sum()/trial.sum()),trace_interval=np.quantile(draws,[.025,.975]).tolist())
        results.append(dict(assumed_fallback_commit_on_copy_opportunity=easy,calibrated_elsewhere=hard,
            matched_four_commits=bool(abs(len(matches)/b.sum()-4)<.005),
            actual_baseline_commits=float(len(matches)/b.sum()),baseline_steps=int(b.sum()),policy_steps=int(steps.sum()),
            copy_verifies=int(hits.sum()),copy_output_tokens=int(counts[:,2].sum()),wide_cost_sensitivity=predictions))
    result=dict(dataset=name,trajectories=n,turns=len(ends),output_tokens=len(matches),tokenizer_sha256=digest,
        frozen_policy="longest of prior same-request matches of 16/8 tokens, require 31 historically available proposal tokens",
        scope="decode-only proxy, known per-turn output budget, skip wide proposals with fewer than 32 remaining, synthetic fallback correlation, no TP effects",
        results=results)
    (ROOT/f"{name}_muse_replay.json").write_text(json.dumps(result,indent=2))
    print(name,n,len(matches),[(v["assumed_fallback_commit_on_copy_opportunity"],v["wide_cost_sensitivity"]["1.25"]["ratio"]) for v in results],flush=True)

if __name__=="__main__": main()
