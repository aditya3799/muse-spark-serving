"""Provider-recorded cache hits as a conditional workload proxy for the 483 public Muse turns."""
import json,statistics
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent/"research_2026-11-18_frontier/patronus_muse.jsonl"
turns=[s["metrics"] for row in P.read_bytes().splitlines() for s in json.loads(row)["steps"] if s["source"]=="agent"]
result={}
for chunk in (8192,16384):
    out={"observed":{},"missing_as_cold":{}}
    for mode in out:
        prompts=hits=omits=benefit=0;uncached=[]
        for m in turns:
            p=m.get("cached_tokens")
            if p is None and mode=="observed":continue
            e=int(m["prompt_tokens"]);p=int(p or 0)
            assert 0<=p<=e
            n=max(0,e-p);skip=0
            for j in range(p,e,chunk):
                end=min(e,j+chunk)
                if j>=2048 and end<=e-2048:skip+=end-j
            hits+=p;prompts+=e;omits+=skip;benefit+=skip>0;uncached.append(n)
        out[mode]=dict(turns=len(uncached),prompt_rows=prompts,provider_cached_rows=hits,new_rows=sum(uncached),row_eligible=omits,turns_eligible=benefit,max_uncached=max(uncached),median_uncached=statistics.median(uncached))
    result[str(chunk)]=out
(H/"public_observed_cache.json").write_text(json.dumps(result,indent=2));print(result)


