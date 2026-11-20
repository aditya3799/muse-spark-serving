"""Bound cross-request attention's possible KV read reuse from actual token prefixes."""
import hashlib
import json
import random
import sys
from pathlib import Path
here=Path(__file__).resolve().parent
root=here.parents[1]
sys.path.insert(0,str(root/"research_2026-11-18_rl_prefill/same_batch_prefix"))
from replay import F,Tokenizer,corpora,initial,lcp

def reduce_case(xs,b):
    repeated=0;visible=0;stress=0;largest=0;requests=0
    for j in range(0,len(xs),b):
        batch=sorted(xs[j:j+b]);requests+=len(batch)
        adjacent=[lcp(a,z) for a,z in zip(batch,batch[1:])]
        repeated+=sum(adjacent);largest=max([largest]+adjacent)
        visible+=sum(map(len,batch))
        # Every prompt keeps the visible initial transcript exactly and then
        # appends a distinct private tail up to the scored 74,160 mean.
        stress+=sum(max(74160,len(v)) for v in batch)
    return dict(requests=requests,possible_duplicate_token_reads=repeated,visible_full_reads=visible,private_tail_full_reads=stress,
        duplicate_visible_fraction=repeated/visible,duplicate_stress_fraction=repeated/stress,longest_common_prefix=largest)
tok=Tokenizer.from_file(str(F/"muse_tokenizer.json"))
report=dict(tokenizer_sha256=hashlib.sha256((F/"muse_tokenizer.json").read_bytes()).hexdigest(),cohorts={})
for key,ds in corpora().items():
    vals=[initial(chat,tok) for chat in ds]
    for order in ("original","permuted"):
        arr=list(vals)
        if order=="permuted":random.Random(923641).shuffle(arr)
        report["cohorts"][key+"/"+order]={str(b):reduce_case(arr,b) for b in (1,4,16,32)}
report["scope"]="Maximum full-context KV duplication assuming every equal logical prefix is physically shared and reused perfectly. Unique private suffixes used only in the 74,160-token stress. No cache miss/DRAM measurements or hidden serving trace. Already cached L2 sharing can only lower the incremental improvement."
(here/"shared_kv.json").write_text(json.dumps(report,indent=2))
print([(k,round(v['32']['duplicate_stress_fraction'],6),v['32']['longest_common_prefix']) for k,v in report["cohorts"].items()])
