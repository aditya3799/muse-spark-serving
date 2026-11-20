"""Reprice the scored parent's full, SWA and draft pools under an unchanged byte budget."""
import json
from pathlib import Path
import re
D=Path(__file__).resolve().parent
j=json.loads((D.parent/"release_2026-11-18/scored_submission_status.json").read_text())
r=j["result"];shape=r["shape"]
raw=r["startup_log"]["text"]
log="\n".join(raw) if isinstance(raw,list) else raw
def pool(name):
    hit=re.search("Allocating "+name+r" hierarchical KV host pool: (\d+) tokens, ([.\d]+) GB",log)
    return int(hit[1]),float(hit[2])
def unit(name):
    n,gb=pool(name)
    return round((gb*1e9/n)/128)*128
full,swa=map(int,re.search(r"Post-capture KV sizing: full_layer_tokens=(\d+), swa_layer_tokens=(\d+)",log).groups())
a,b,c=[unit(k) for k in ("full","swa","draft")]
assert (a,b,c)==(4096,11776,2560)
budget=full*(a+c)+swa*b
last=shape["first_turn"]+(shape["turns"]-1)*(shape["new_per_turn"]+shape["output_mean"])
rows=[]
for ratio in (.8,.5,.375,.35,.3):
    n=int(budget/(a+c+ratio*b))
    rows.append(dict(ratio=ratio,full_tokens=n,swa_tokens=int(ratio*n),average_last_turn_full_residency=int(n/last),
                     max_full_token_margin_at_32=int(n-32*last)))
data=dict(job=j["id"],per_token_bytes=dict(full=a,swa=b,draft=c),current=dict(full=full,swa=swa,budget_bytes=budget),
          first_prompt_tokens=shape["first_turn"],mean_last_prompt_tokens=last,ratios=rows,
          observed_tok_s_per_gpu=[x["tok_s_per_gpu"] for x in r["rungs"]],
          scope="Capacity-only and mean lengths. Reallocation assumes constant available bytes, page alignment negligible, separate draft pool, the same graph reservations, and sufficient SWA residency. This is not a wall-time measurement.")
(D/"cache_capacity.json").write_text(json.dumps(data,indent=2))
print(rows)


