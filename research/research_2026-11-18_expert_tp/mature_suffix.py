"""Causal suffix retrieval with maturity-specific indexes. Exact local corpus replay."""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from numba import njit
from tokenizers import Tokenizer
HERE=Path(__file__).resolve().parent
FRONTIER=HERE.parent/"research_2026-11-18_frontier"
sys.path.insert(0,str(FRONTIER))
from cross_validate_lookup import dataset, per_turn

WIDTH=(8,16,32,64)
POLICIES=("neural8","old32","mature32","adaptive","adaptive_unknown_remaining")
RULES=((16,64),(16,32),(8,32),(16,16),(8,16))

def common_prefix(raw, src, dst, cap):
    # Byte slices test exact equality of little-endian u32 tokens; no hash collisions.
    lo,hi=0,cap
    while lo<hi:
        mid=(lo+hi+1)//2
        if raw[src*4:(src+mid)*4]==raw[dst*4:(dst+mid)*4]: lo=mid
        else: hi=mid-1
    return lo

def replay_messages(messages, tok):
    encoded=[]
    for m in messages:
        ids=tok.encode(m["content"],add_special_tokens=False).ids
        encoded.append((m["role"],ids))
    yield from replay_ids(encoded)

def replay_ids(encoded):
    groups=[]
    seq=array("i")
    assert seq.itemsize==4
    for role,ids in encoded:
        seq.append(-1)
        begin=len(seq)
        seq.extend(ids)
        groups.append((role,begin,len(seq)))
    raw=seq.tobytes()
    latest={8:{},16:{}}
    eligible={rule:{} for rule in RULES}
    for role,begin,stop in groups:
        n=stop-begin
        if n==0: continue
        assist=role=="assistant"
        if assist:
            # cols: old32, mature32, adaptive, adaptive ignoring recorded remainder
            widths=np.full((n,4),8,np.uint8)
            prefixes=np.zeros((n,4),np.uint8)
        for pos in range(n+1):
            now=begin+pos
            for k,w in RULES:
                if pos < k+w-1: continue
                # Exactly w-1 continuation tokens in THIS message have arrived.
                old=now-(w-1)
                key=raw[(old-k)*4:old*4]
                eligible[k,w][key]=old
            if pos==n: break
            keys={k:raw[(now-k)*4:now*4] for k in (16,8) if pos>=k}
            if assist and keys:
                rem=n-pos
                # Mirror original prior strategy exactly: the most recent hit at
                # the longest key wins even when its continuation is too short.
                if rem>=32:
                    for k,key in keys.items():
                        old=latest[k].get(key)
                        if old is not None:
                            src,msg_end=old
                            if min(msg_end,now)-src>=31:
                                widths[pos,0]=32
                                prefixes[pos,0]=common_prefix(raw,src,now,31)
                            break
                matched_cache={}
                for k,w in RULES:
                    key=keys.get(k)
                    if key is None: continue
                    src=eligible[k,w].get(key)
                    if src is None: continue
                    assert src+w-1<=now
                    for col,take in ((1,w==32 and rem>=32),
                                     (2,rem>=w),(3,True)):
                        if widths[pos,col]!=8 or not take: continue
                        widths[pos,col]=w
                        cache_key=(src,w)
                        if cache_key not in matched_cache:
                            matched_cache[cache_key]=common_prefix(raw,src,now,min(w-1,rem))
                        prefixes[pos,col]=matched_cache[cache_key]
            for k,key in keys.items():
                latest[k][key]=(now,stop)
        if assist: yield widths,prefixes

@njit
def run(widths,prefixes,mode):
    # For width 8 synthetic neural commits are the very same four at every position.
    stats=np.zeros((4,4),np.int64)
    i=0
    while i<len(widths):
        w=8 if mode==0 else int(widths[i,mode-1])
        matched=0 if w==8 else int(prefixes[i,mode-1])
        moved=min(len(widths)-i,4 if w==8 else matched+1)
        b=0 if w==8 else 1 if w==16 else 2 if w==32 else 3
        stats[b,0]+=1
        stats[b,1]+=moved
        stats[b,2]+=min(moved,matched)
        stats[b,3]+=w
        i+=moved
    return stats

def adversarial_test():
    x=list(range(100,116)); old_cont=list(range(300,364))
    history=x+old_cont+[-8]+x+[9,8,7]
    # Two previous messages with same key; the newer occurrence has no mature tail.
    enc=[("tool",history[:80]),("user",history[81:]),
         ("assistant",x+old_cont+[10]*20)]
    wid,mat=list(replay_ids(enc))[0]
    assert wid[16].tolist()==[8,32,64,64] and mat[16].tolist()==[0,31,63,63]
    wid,mat=list(replay_ids([("assistant",x+old_cont)]))[0]
    assert np.all(wid==8)
    # The last exact k-gram of a message cannot source a continuation from the next.
    wid,mat=list(replay_ids([("user",x),("tool",old_cont),("assistant",x+old_cont)]))[0]
    assert np.all(wid[16]==8)
    # At a recorded tail, adapt to 16 for 20 available future outputs, unknown-cap path still pays 64.
    wid,mat=list(replay_ids([("user",x+old_cont),("assistant",x+old_cont[:20])]))[0]
    assert wid[16].tolist()==[8,8,16,64] and mat[16].tolist()==[0,0,15,20]

def summarize(t,baseline):
    rounds=int(t[:,0].sum())
    wide=rounds-int(t[0,0])
    rows=int(t[:,3].sum())
    def estimate(c16,c32,c64):
        return baseline/(t[:,0]@np.array([1.,c16,c32,c64]))
    slack=baseline-rounds
    coefficient=rows/8-rounds
    return dict(rounds=rounds,committed_tokens=int(t[:,1].sum()),target_rows=rows,
        by_width={str(w):dict(calls=int(x[0]),commits=int(x[1]),matched_copy_proposals=int(x[2]),target_rows=int(x[3])) for w,x in zip(WIDTH,t)},
        copied_proposal_tokens=int(t[:,2].sum()),copy_commits=int(t[1:,1].sum()),
        equal_cost_round_gain=baseline/rounds,target_row_ratio=rows/(8*baseline),
        common_wide_cost_break_even=(1+slack/wide if wide else None),
        marginal_row_cost_break_even=(slack/coefficient if coefficient else None),
        ratios_for_width_costs={"1,1,1":estimate(1,1,1),"1.125,1.375,1.875":estimate(1.125,1.375,1.875),
           "1.25,1.75,2.75":estimate(1.25,1.75,2.75),"1.5,2.5,4.5":estimate(1.5,2.5,4.5),
           "2,4,8":estimate(2,4,8)})

def main():
    parser=argparse.ArgumentParser();parser.add_argument("corpus",choices=("swesmith","openhands"))
    name=parser.parse_args().corpus
    adversarial_test()
    path=FRONTIER/"muse_tokenizer.json"
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha=="ac2f41ef2444dec82fdc90e334f80543f298535005fe78aec7e8311ef86f84d0"
    tok=Tokenizer.from_file(str(path))
    totals=np.zeros((5,4,4),np.int64)
    turns=tokens=old_positions=0
    per_req=[]
    for req,msgs in enumerate(dataset(name)):
        counts=np.zeros_like(totals)
        reference=iter(per_turn(msgs,tok)) if req<4 else None
        for widths,match in replay_messages(msgs,tok):
            n=len(widths); tokens+=n; turns+=1
            if reference is not None:
                r=next(reference); assert len(r)==n
                for pos,(k,m,available) in enumerate(r):
                    old=(k>=8 and available>=31 and n-pos>=32)
                    assert int(widths[pos,0])==(32 if old else 8)
                    assert not old or match[pos,0]==m
                old_positions+=n
            for mode in range(5): counts[mode]+=run(widths,match,mode)
        if reference is not None: assert next(reference,None) is None
        totals+=counts
        per_req.append(counts[:,:,0].tolist())
    base=int(totals[0,:,0].sum())
    results={p:summarize(totals[m],base) for m,p in enumerate(POLICIES)}
    assert all(r["committed_tokens"]==tokens for r in results.values())
    documented=json.loads((FRONTIER/f"{name}_muse_replay.json").read_bytes())
    assert (tokens,turns,len(per_req))==(documented["output_tokens"],documented["turns"],documented["trajectories"])
    out=dict(corpus=name,requests=len(per_req),turns=turns,output_tokens=tokens,tokenizer_sha256=sha,
        original_policy_exactly_crosschecked_positions=old_positions,
        method="Fresh index per trajectory. Only within-message keys and continuations. Lookup candidate selected before reading responses. Old policy mirrors original longest-most-recent 16/8 width32. Mature32 uses most recent fully mature 16/8 key. Adaptive prioritizes (suffix,width) (16,64), (16,32), (8,32), (16,16), (8,16), else neural8. Fixed four-token synthetic neural commit with turn-end truncation. Each known-cap mode respects remaining recorded output, an optimistic offline assumption. Unknown-remaining spends full width through EOS. No DFlash model, hardware timing, padding, draft work or score.",
        break_even="Sum of n_width * cost_width must be less than baseline rounds. Common wide cost assigns all wide verifications one multiplier. Marginal row cost sets cost_w=1+beta*(w/8-1), in units of an 8-row verify. Grid keys give 16,32,64 costs in that order.",
        results=results,per_request_rounds_by_width=per_req)
    (HERE/f"mature_suffix_{name}.json").write_text(json.dumps(out,separators=(",",":"))+"\n")
    print(name,tokens,[(p,r["rounds"],r["equal_cost_round_gain"],r["target_row_ratio"]) for p,r in results.items()])
if __name__=="__main__": main()

