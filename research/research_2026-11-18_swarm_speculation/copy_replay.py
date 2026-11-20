"""Causal own-request suffix replay on public SWE-smith agent trajectories.

External text is used solely to evaluate, never as a source of proposals. A
history index discards all other request histories between traces.
"""
import collections
import hashlib
import json
import math
import random
from pathlib import Path
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
KEYS = (4, 8, 16)
WIDTHS = (8, 16, 32)
GATES = (4, 8, 16)
# Policy is chosen on the ten earliest pilot rows against a prespecified 1.25x
# cost per extra-wide verify, then frozen before evaluating forty held-out rows.
PILOT_COST = 1.25

def per_turn(messages, tokenizer):
    index = {k:{} for k in KEYS}
    history = []
    for msg in messages:
        role = msg["role"]
        # A unique separator prevents proposals crossing a tool or message boundary.
        history.append(-1)
        ids = tokenizer.encode(msg["content"], add_special_tokens=False).ids
        rows = []
        since = 0
        for pos,tok in enumerate(ids):
            end = len(history)
            if role == "assistant":
                chosen = None
                for k in reversed(KEYS):
                    if since >= k:
                        token_key = tuple(history[end-k:end])
                        old = index[k].get(token_key)
                        if old is not None and old < end:
                            chosen = (k,old)
                            break
                # Selection is entirely determined before reading the emitted token.
                # Replay compares the proposal with the actual output only after selection.
                prefix, matched, available = 0,0,0
                if chosen:
                    prefix,old=chosen
                    cap=min(31,end-old)
                    while available<cap and history[old+available]>=0:
                        available+=1
                    while matched<min(available,len(ids)-pos) and history[old+matched]==ids[pos+matched]:
                        matched+=1
                rows.append((prefix,matched,available))
            for k in KEYS:
                if since>=k:
                    index[k][tuple(history[end-k:end])] = end
            history.append(tok)
            since+=1
        if rows:
            yield rows

def simulate(rows,base,width,gate):
    i=steps=hits=hit_tokens=verify_rows=0
    while i<len(rows):
        prefix,matched,available=rows[i]
        # Deploy only a full proposed block; no reading target outputs to gate.
        if prefix>=gate and available>=width-1:
            moved=min(len(rows)-i,min(width,1+matched))
            hits+=1
            hit_tokens+=moved
            verify_rows+=width
        else:
            moved=min(len(rows)-i,base)
            verify_rows+=8
        steps+=1
        i+=moved
    return (steps,hits,hit_tokens,verify_rows)

def oracle_steps(rows,base,width):
    # Upper bound on eliminating verifies for this *fixed family* of proposals.
    # Optimizer sees future acceptance to choose copy or DFlash; deployable policy cannot.
    n=len(rows)
    best=[0]*(n+1)
    for i in range(n-1,-1,-1):
        prefix,matched,available=rows[i]
        fallback=best[min(n,i+base)]
        copy=best[min(n,i+1+min(width-1,matched))] if prefix and available>=width-1 else fallback
        best[i]=1+min(copy,fallback)
    return best[0]

def new_stat():
    return collections.Counter()

def replay(trace,tok):
    stats=collections.defaultdict(new_stat)
    for rows in per_turn(json.loads(trace["messages"]),tok):
        for base in (2,4,6):
            plain=math.ceil(len(rows)/base)
            for width in WIDTHS:
                ideal=oracle_steps(rows,base,width)
                for gate in GATES:
                    steps,hits,copied,vr=simulate(rows,base,width,gate)
                    stats[base,width,gate].update(turns=1,generated=len(rows),baseline=plain,
                        steps=steps,hits=hits,copied=copied,verify_rows=vr,oracle_steps=ideal)
    return stats

def summary(x):
    out=dict(x)
    out["ideal_equal_cost_ratio"]=x["baseline"]/x["steps"]
    out["clairvoyant_equal_cost_ratio"]=x["baseline"]/x["oracle_steps"]
    out["wide_to_fallback_cost_break_even"]=1+(x["baseline"]-x["steps"])/max(1,x["hits"])
    out["ratio_at_1p25_cost"]=x["baseline"]/(x["steps"]+.25*x["hits"])
    out["relative_target_rows"]=x["verify_rows"]/(8*x["baseline"])
    out["hit_step_fraction"]=x["hits"]/x["steps"]
    return out

def read_and_run():
    tokenizer=Tokenizer.from_file(str(HERE/"qwen_proxy.json"))
    sets={"pilot":[],"holdout":[]}
    hashes={}
    for path in sorted(HERE.glob("traces_*.json")):
        rows=json.loads(path.read_text(encoding="utf8"))["rows"]
        assert len(rows)==10 and all(not r.get("truncated_cells") for r in rows)
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        stage="pilot" if path.stem=="traces_0" else "holdout"
        for r in rows:
            sets[stage].append((r["row"]["instance_id"],replay(r["row"],tokenizer)))
    assert len({name for split in sets.values() for name,_ in split})==50
    def merge(group):
        d=collections.defaultdict(new_stat)
        for _,stats in group:
            for k,v in stats.items(): d[k].update(v)
        return d
    pilot,holdout=merge(sets["pilot"]),merge(sets["holdout"])
    def ratio(x): return x["baseline"]/(x["steps"]+(PILOT_COST-1)*x["hits"])
    best=max((key for key in pilot if key[0]==4),key=lambda key:ratio(pilot[key]))
    rng=random.Random(4519)
    sampled=[ratio(merge(rng.choices(sets["holdout"],k=len(sets["holdout"])))[best]) for _ in range(400)]
    sampled.sort()
    result=dict(dataset="SWE-bench/SWE-smith-trajectories, default, ticks",
      offsets=[0,1000,5000,12000,20000],rows_per_offset=10,pilot=10,heldout=40,
      tokenizer="Qwen/Qwen2.5-Coder-7B, proxy",tokenizer_sha256=hashlib.sha256((HERE/"qwen_proxy.json").read_bytes()).hexdigest(),
      hashes=hashes,selected=dict(base=best[0],width=best[1],gate=best[2]),
      pilot_selected=summary(pilot[best]),heldout_selected=summary(holdout[best]),
      heldout_trace_bootstrap_ratio_1p25_95=[sampled[10],sampled[-11]],
      heldout_by_policy={str(key):summary(val) for key,val in holdout.items()},
      per_trajectory={name:dict(s[best]) for name,s in sets["holdout"]})
    (HERE/"copy_replay.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k in ("selected","pilot_selected","heldout_selected","heldout_trace_bootstrap_ratio_1p25_95")},indent=2))

if __name__=="__main__":
    read_and_run()
