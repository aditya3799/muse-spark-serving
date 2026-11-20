"""Selective cold-first prefill replay with observed real Muse text and official-shape scenarios."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

from policies import RepairLedger,repair_omissions,window_floor,removable_floor
HERE=Path(__file__).resolve().parent
PUBLIC=HERE.parent/"research_2026-11-18_frontier"
sys.path.insert(0,str(PUBLIC))
from muse_records import samples


def provider():
    for line in (PUBLIC/"patronus_muse.jsonl").read_bytes().splitlines():
        turns=[s["metrics"] for s in json.loads(line)["steps"] if s["source"]=="agent"]
        yield [(m["prompt_tokens"],m["completion_tokens"]) for m in turns]


def visible():
    from tokenizers import Tokenizer
    src=PUBLIC/"muse_tokenizer.json"
    assert hashlib.sha256(src.read_bytes()).hexdigest()=="ac2f41ef2444dec82fdc90e334f80543f298535005fe78aec7e8311ef86f84d0"
    tok=Tokenizer.from_file(str(src))
    for trace in samples(joined=True):
        n=0
        result=[]
        for message in trace:
            n+=1
            count=len(tok.encode(message["content"],add_special_tokens=False).ids)
            if message["role"]=="assistant": result.append((n,count))
            n+=count
        yield result


def scenario(trace,policy,chunk,window,retention):
    cache={}
    # Only logical positions matter for this single common-prefix trace.
    key=range(64)
    ledger=RepairLedger(window)
    s=collections.Counter()
    for end,out in trace:
        cutoff=window_floor(end,window)
        if policy=="native": cls=0
        elif policy=="floor": cls=cutoff
        elif policy=="inherit":
            candidates=[c for c in cache if c<=cutoff]
            cls=max(candidates,key=lambda c:cache[c]) if candidates else cutoff
        else: cls=0
        ordinary=min(end-1,cache.get(cls,0))
        if policy=="repair":
            hit=ledger.limit(key,end,ordinary)
            omitted=0
            for start in range(hit,end,chunk):
                stop=min(end,start+chunk)
                if start>=ledger.key_length and stop<=end-window:
                    # Use an indexable length-only prefix with the shared real key.
                    ledger.mark(range(end),start,stop)
                    omitted+=stop-start
            s["bounded_replay"]+=ordinary-hit
            publish=ledger.publish(key,end+out)
        else:
            hit=ordinary
            omitted=0 if policy=="native" else removable_floor(hit,end,chunk,cls,window)
            publish=True
        s["prefill_rows"]+=end-hit
        s["draft_prefill_rows"]+=end-hit-omitted
        s["omitted"]+=omitted
        s["turns_with_omission"]+=int(bool(omitted))
        s["output"]+=out
        s["turns"]+=1
        if publish and retention!="cold":
            cache[cls]=max(cache.get(cls,0),end+(out if retention=="continuation" else 0))
        if not publish: s["private_tips"]+=1
    return s


def main():
    data={"recorded_provider":list(provider()),"visible_exact_tokenizer":list(visible()),
          "official_shape_13_turns":[[(74160+i*(753+220),220) for i in range(13)]]}
    result={}
    for source,traces in data.items():
        result[source]={}
        for chunk in (8192,16384):
            for retention in ("continuation","prompt_only","cold"):
                out={}
                for policy in ("native","floor","inherit","repair"):
                    metrics=sum((scenario(t,policy,chunk,2048,retention) for t in traces),collections.Counter())
                    out[policy]=dict(metrics)
                result[source][str(chunk)+"/"+retention]=out
    (HERE/"replay.json").write_text(json.dumps(result,indent=2)+"\n")
    for source,tests in result.items():
        for k in ("8192/continuation","16384/continuation"):
            a=tests[k]
            print(source,k,"native",a["native"]["prefill_rows"],
                  [(p,z["prefill_rows"]-a["native"]["prefill_rows"],z.get("omitted",0)) for p,z in a.items() if p!="native"])
if __name__=="__main__":main()

