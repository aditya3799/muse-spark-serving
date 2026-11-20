"""Batch-policy sensitivity on held-out public trajectories with the exact tokenizer.

Uses continuous refill of completed assistant turns in deterministic shuffled order.
All predictors see only their own confirmed history. Independent turns are scheduling
units; input-service time, H200 width32 costs and live SGLang clocks are absent.
"""
import collections
import hashlib
import json
from pathlib import Path
import random
from tokenizers import Tokenizer
from copy_replay import per_turn

HERE=Path(__file__).resolve().parent
def run(turns,batch,policy):
    next_turn=0
    group=[]
    totals=collections.Counter()
    while group or next_turn<len(turns):
        while len(group)<batch and next_turn<len(turns):
            group.append([next_turn,0]);next_turn+=1
        flags=[]
        for idx,i in group:
            k,_,avail=turns[idx][i]
            flags.append(k>=8 and avail>=31 and len(turns[idx])-i>=32)
        b=len(group)
        potential=sum(flags)
        if policy=="base" or policy=="uniform" and potential<b:
            flags=[False]*b
        hits=sum(flags)
        turns_done=[]
        for j,((idx,i),hit) in enumerate(zip(group,flags)):
            step=1+min(turns[idx][i][1],31) if hit else 4
            step=min(len(turns[idx])-i,step)
            group[j][1]+=step
            totals["generated"]+=step
            if hit:totals["copy_tokens"]+=step
            if group[j][1]==len(turns[idx]):turns_done.append(j)
        for j in reversed(turns_done): group.pop(j)
        totals["batch_rounds"]+=1
        totals["target_forwards"]+=2 if policy=="split" and 0<hits<b else 1
        totals["normal_draft_forwards"]+=int(hits<b)
        totals["eligible_rows"]+=potential
        totals["active_rows"]+=b
        totals["copy_rows"]+=hits
        totals["uniform_possible_batches"]+=int(potential==b)
        totals["target_rows"]+=8*b+24*hits
        totals["wide_or_mixed_batches"]+=int(hits>0)
    return totals

def main():
    tokenizer_path=HERE.parent/"research_2026-11-18_frontier/muse_tokenizer.json"
    sha=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    assert sha=="ac2f41ef2444dec82fdc90e334f80543f298535005fe78aec7e8311ef86f84d0"
    tok=Tokenizer.from_file(str(tokenizer_path))
    turns=[]
    for off in (1000,5000,12000,20000):
        for row in json.loads((HERE/f"traces_{off}.json").read_text(encoding="utf8"))["rows"]:
            turns.extend(per_turn(json.loads(row["row"]["messages"]),tok))
    output={"tokenizer_sha256":sha,"turns":len(turns),"seeds":{}}
    for seed in (109,343,891):
        order=turns.copy()
        random.Random(seed).shuffle(order)
        counts={}
        for b in (1,4,16,32):
            baseline=run(order,b,"base")
            variants={}
            for policy in ("uniform","split","ragged"):
                val=run(order,b,policy)
                assert val["generated"]==baseline["generated"]
                variants[policy]=dict(val)
                variants[policy].update(target_forward_ratio=baseline["target_forwards"]/val["target_forwards"],
                   batch_round_ratio=baseline["batch_rounds"]/val["batch_rounds"],
                   target_row_ratio=val["target_rows"]/baseline["target_rows"])
            counts[b]={"base":dict(baseline),**variants}
        output["seeds"][seed]=counts
    (HERE/"batch_replay.json").write_text(json.dumps(output,indent=2))
    for b,group in output["seeds"][109].items():
        print(b,[(p,round(x["target_forward_ratio"],4),round(x["target_row_ratio"],4),x["wide_or_mixed_batches"]) for p,x in group.items() if p!="base"])
if __name__=="__main__":
    main()
