"""Replay prompt-token admission, testing the exact proposed scheduler method."""
from array import array
from collections import Counter
import ast
import hashlib
import json
import random
import statistics
from pathlib import Path
from types import SimpleNamespace
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from build import HERE, ROOT, METHOD

F = ROOT / "research_2026-11-18_frontier"
S = ROOT / "research_2026-11-18_swarm_speculation"
data = {}
exec("class Candidate:\n" + METHOD, {"Req":object,"IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD":32}, data)
candidate = data["Candidate"]()
candidate.tree_cache = SimpleNamespace(supports_mamba=lambda:False, supports_prefix_sharing=lambda:True)
candidate.page_size = 1

def lcp(a,b):
    left, right = 0, min(len(a),len(b))
    while left < right:
        mid=(left+right+1)//2
        if a[:mid]==b[:mid]: left=mid
        else: right=mid-1
    return left

def run(prompts, cap, mode, oracle=False, lag=0, initial=None):
    n=len(prompts)
    common=[[len(a) if a is b else lcp(a,b) for b in prompts] for a in prompts]
    progress=list(initial or [0]*n)
    published=progress[:]
    pending=None
    waiting=list(range(n))
    active=None
    done={}
    total=steps=deferred=0
    cache={} if oracle else None
    while waiting or active is not None:
        steps+=1
        assert steps < 10+sum(map(len,prompts))
        candidate.can_run_list=[]
        budget=cap
        admitted=[]
        order=([active] if active is not None else [])+waiting[:]
        for i in order:
            if not budget: break
            ids=prompts[i]
            if i==active: hit=progress[i]
            else: hit=min(len(ids)-1,max(min(common[i][j],p) for j,p in enumerate(published)))
            req=SimpleNamespace(full_untruncated_fill_ids=ids,prefix_len=hit,last_node=bytes(ids[:hit]),
                                extra_key=None,cache_salt=None,host_hit_length=0,skip_radix_cache_insert=False,
                                is_dllm=lambda:False,return_logprob=False)
            candidate.rem_chunk_tokens=budget
            if mode=='candidate' and i!=active and candidate.shares_admitted_prefix(req):
                deferred+=1
                continue
            end=min(hit+budget,len(ids))
            req.extend_end=end
            candidate.can_run_list.append(req)
            admitted.append((i,hit,end))
            budget-=end-hit
            total+=end-hit
        assert admitted
        active=None
        for i,hit,end in admitted:
            if i in waiting: waiting.remove(i)
            if end!=len(prompts[i]):
                assert active is None
                active=i
            else: done[i]=end
            if oracle:
                assert hit==0 or tuple(prompts[i][:hit]) in cache
                state=cache[tuple(prompts[i][:hit])] if hit else 11
                for pos in range(hit,end):
                    state=(state*1000003+prompts[i][pos]+pos)%1000000007
                    key=tuple(prompts[i][:pos+1])
                    assert cache.setdefault(key,state)==state
                if i in done: done[i]=state
            progress[i]=end
        if lag and pending is not None:
            published=pending
        elif not lag:
            published=progress[:]
        pending=progress[:]
    assert len(done)==n
    return dict(rows=total,batches=steps,deferred=deferred),done

def corpora():
    with (F/"patronus_muse.jsonl").open("rb") as handle:
        muse=[[{"role":"user","content":json.loads(line)["steps"][0]["message"]}] for line in handle]
    opens=[r["messages"] for r in pq.read_table(F/"openhands.parquet",columns=["messages"]).to_pylist()]
    smith=[]
    for off in (1000,5000,12000,20000):
        rows=json.loads((S/f"traces_{off}.json").read_text(encoding="utf8"))["rows"]
        smith.extend(json.loads(r["row"]["messages"]) for r in rows)
    return {"muse":muse,"openhands":opens,"swesmith":smith}

def initial(chat,tok):
    ids=array("q")
    for m in chat:
        if m["role"]=="assistant": break
        ids.extend(tok.encode(m["role"]+"\n"+(m["content"] or "")+"\n",add_special_tokens=False).ids)
    return ids

def summary(prompts,cap=16384,lag=0,initial=None):
    out={}
    for load in (1,4,16,32):
        stats={}
        for name in ("baseline","candidate"):
            agg=Counter()
            for start in range(0,len(prompts),load):
                batch=prompts[start:start+load]
                known=initial[start:start+load] if initial is not None else None
                ret,_=run(batch,cap,name,lag=lag,initial=known)
                agg.update(ret)
            stats[name]=dict(agg)
        stats["saved_fraction"]=1-stats["candidate"]["rows"]/stats["baseline"]["rows"]
        out[str(load)]=stats
    return out

def guards():
    ids=array("q",range(192))
    def req():
        return SimpleNamespace(full_untruncated_fill_ids=ids,prefix_len=16,last_node=8,extend_end=160,
            extra_key="adapter",cache_salt="scope",host_hit_length=0,skip_radix_cache_insert=False,
            is_dllm=lambda:False,return_logprob=False)
    candidate.page_size=16
    candidate.rem_chunk_tokens=128
    candidate.can_run_list=[req()]
    assert candidate.shares_admitted_prefix(req())
    gates={"extra_key":"different","cache_salt":"different","last_node":9,"prefix_len":15,"host_hit_length":16,
        "skip_radix_cache_insert":True,"is_dllm":lambda:True,"return_logprob":True}
    for field,value in gates.items():
        x=req(); setattr(x,field,value)
        assert not candidate.shares_admitted_prefix(x)
    other=req(); other.full_untruncated_fill_ids=ids[:]; other.full_untruncated_fill_ids[55]+=1
    assert not candidate.shares_admitted_prefix(other)
    candidate.page_size=1
    for seed in range(50):
        rng=random.Random(seed)
        template=array("q",rng.choices(range(40),k=180))
        cases=[template[:rng.randint(0,len(template))]+array("q",rng.choices(range(40),k=rng.randint(1,60))) for _ in range(12)]
        assert run(cases,160,"candidate",True)[1]==run(cases,160,"baseline",True)[1]
    return {"namespaces_and_guards":10,"causal_cache_differential_seeds":50,"requests_per_seed":12}

def main():
    tests=guards()
    tfile=F/"muse_tokenizer.json"
    assert hashlib.sha256(tfile.read_bytes()).hexdigest()=="ac2f41ef2444dec82fdc90e334f80543f298535005fe78aec7e8311ef86f84d0"
    tok=Tokenizer.from_file(str(tfile))
    result={}
    for name,chats in corpora().items():
        prompts=[initial(m,tok) for m in chats]
        result[name]={"requests":len(prompts),"median_visible_input":statistics.median(map(len,prompts)),
                      "measured_visible_parts":summary(prompts)}
        # Judge-length stress keeps the observed prompt common prefixes. The added tails are distinct.
        padded=[p+array("q",[1000000+i])*(max(0,74160-len(p))) for i,p in enumerate(prompts[:32])]
        result[name]["distinct_tail_74160_stress"]=summary(padded)
        result[name]["delayed_publication_visible_parts"]=summary(prompts,lag=1)
        result[name]["delayed_publication_74160_stress"]=summary(padded,lag=1)
    # Each independent conversation reuses its own committed branch. Distinct keys
    # stand in for the 74K prefix, since an exact distinct prefix cannot rejoin.
    later=[array("q",[900000+i])*64+array("q",[42])*753 for i in range(32)]
    for lag in (0,1):
        result["later_distinct_committed_branches_lag"+str(lag)]=summary(later,lag=lag,initial=[64]*len(later))
    identical=[array("q",[7])*74160 for _ in range(32)]
    result["identical_positive_control"]=summary(identical)
    result["identical_delayed_positive_control"]=summary(identical,lag=1)
    output={"baseline_sha256":"8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b",
        "tests":tests,"replay":result,
        "limits":"Cohorts arrive together and start with an empty radix. Publication is modeled both immediately and one admission pass late. Repeated later turns use distinct already-committed branches and a 753-token extension. The 64-token unique keys stand in for the private canonical long prefixes. Tokenization uses visible role/content fields, not a server chat-template. No judge token IDs or per-request arrival times are available. 74160 is the official mean, not a recorded individual request. No hardware latency, decode or TP is simulated."}
    (HERE/"replay.json").write_text(json.dumps(output,indent=2),encoding="utf8")
    for name,d in result.items():
        table=d.get("measured_visible_parts",d)
        print(name,{k:(round(v["saved_fraction"],5),v["baseline"]["batches"],v["candidate"]["batches"]) for k,v in table.items()})
        if "distinct_tail_74160_stress" in d:
            print("judge-length", {k:round(v["saved_fraction"],6) for k,v in d["distinct_tail_74160_stress"].items()})
if __name__=="__main__": main()

