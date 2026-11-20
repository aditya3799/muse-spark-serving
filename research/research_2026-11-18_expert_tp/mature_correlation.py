"""Synthetic positive correlation stress test for the frozen maturity policy."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numba import njit
from tokenizers import Tokenizer
from mature_suffix import dataset, FRONTIER, HERE, replay_messages, summarize, POLICIES, WIDTH

@njit
def walk(widths,prefixes,easy_positions,uniform,ends,mode,easy,hard):
    stats=np.zeros((4,4),np.int64)
    i=0
    for stop in ends:
        while i<stop:
            w=8 if mode==0 else widths[i,mode-1]
            matched=0 if w==8 else prefixes[i,mode-1]
            fallback=easy if easy_positions[i] else hard
            normal=int(fallback)+(uniform[i]<(fallback-int(fallback)))
            moved=min(stop-i,normal if w==8 else matched+1)
            b=0 if w==8 else 1 if w==16 else 2 if w==32 else 3
            stats[b,0]+=1
            stats[b,1]+=moved
            stats[b,2]+=min(moved,matched)
            stats[b,3]+=w
            i+=moved
    return stats

def main():
    p=argparse.ArgumentParser();p.add_argument("corpus",choices=("swesmith","openhands"));name=p.parse_args().corpus
    src=FRONTIER/"muse_tokenizer.json"
    digest=hashlib.sha256(src.read_bytes()).hexdigest()
    assert digest=="ac2f41ef2444dec82fdc90e334f80543f298535005fe78aec7e8311ef86f84d0"
    tok=Tokenizer.from_file(str(src))
    ws,ms=[],[]
    for messages in dataset(name):
        for w,m in replay_messages(messages,tok):
            ws.append(w);ms.append(m)
    ends=np.cumsum([len(w) for w in ws]);ws=np.concatenate(ws);ms=np.concatenate(ms)
    # Ease is itself causal and unchanged across ALL trials. It indicates that the
    # current suffix had ANY completed qualifying continuation before the request's
    # current position, independent of actual next tokens or eventual turn length.
    easy_positions=ws[:,3]!=8
    uniform=np.random.default_rng(20261118).random(len(ws),dtype=np.float32)
    records=[]
    for easy in (4.,6.,8.):
        lo,hi=1.,8.
        tried=[]
        for _ in range(20):
            hard=(lo+hi)/2
            baseline=walk(ws,ms,easy_positions,uniform,ends,0,easy,hard)
            committed=int(baseline[:,1].sum()); rounds=int(baseline[:,0].sum())
            mean=committed/rounds
            tried.append((abs(mean-4),hard,mean,rounds))
            if mean<4: lo=hard
            else: hi=hard
        err,hard,mean,rounds=min(tried)
        out={}
        for mode in (0,1,2,3,4):
            t=walk(ws,ms,easy_positions,uniform,ends,mode,easy,hard)
            assert int(t[:,1].sum())==len(ws)
            out[POLICIES[mode]]=summarize(t,rounds)
        records.append(dict(easy=easy,hard=hard,mean_neural_accept=mean,mean_error=err,results=out))
    result=dict(corpus=name,output_tokens=len(ws),tokenizer_sha256=digest,turns=len(ends),
       easy_position_fraction=float(easy_positions.mean()),assumptions="Identical seeded per-token neural lengths for all policies within a scenario. Easy is matured causal >=8-token suffix with any complete historical continuation of >=15 tokens, independent of future and EOS. Easy positions receive fixed 4/6/8 neural commits; other positions have a shared fractional mean calibrated by actual traversals to mean four overall. Synthetic correlation sensitivity; no neural logits, scheduled batches or wall time.",
       records=records)
    (HERE/f"mature_correlation_{name}.json").write_text(json.dumps(result,indent=2)+"\n")
    print(name,[(r["easy"],r["hard"],r["mean_neural_accept"],r["results"]["adaptive"]["ratios_for_width_costs"]["1.25,1.75,2.75"]) for r in records])
if __name__=="__main__": main()

