"""Compare sticky, sticky promotion, and prefix-region omission under adversarial order."""
from array import array
from collections import Counter
import hashlib,json,random
from pathlib import Path
from types import SimpleNamespace as S
from namespaces import DFlashWindowNamespaces
from regions import DFlashRegionCoverage
from policies import window_floor
from safety import ToyCache
H=Path(__file__).resolve().parent
W,C=2048,16384
def request(ids,salt):
    return S(origin_input_ids=ids,output_ids=[],full_untruncated_fill_ids=ids,session=None,multimodal_inputs=None,
        positional_embed_overrides=None,skip_radix_cache_insert=False,extra_key=None,cache_salt=salt)
def common(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y:return i
    return min(len(a),len(b))
def experiments(trace,page,policy):
    ns=DFlashWindowNamespaces(W,page,1)
    reg=DFlashRegionCoverage(W,page)
    bank={};old={};stats=Counter();records=[]
    for ids,salt in trace:
        r=request(ids,salt)
        cutoff=0
        if policy=="sticky":ns.assign(r)
        elif policy=="promote":
            proposed=window_floor(len(ids),W)
            # Store at most one positive preferred floor for this salt + positive-length stem.
            key=(salt,tuple(ids[:max(page,64)]))
            prev=old.get(key)
            floor=prev if prev is not None and prev<=proposed else proposed
            if proposed and prev is None:old[key]=proposed
            r._draft_window_floor=floor;r.cache_salt=json.dumps([salt,floor])
        q=bank.setdefault(r.cache_salt,[])
        ordinary=min(len(ids)-1,max((common(ids,x) for x in q),default=0))
        hit=ordinary if policy!="regions" else reg.match_limit(r,ids,ordinary)
        hit=hit//page*page; omitted=0;published=0
        for p in range(hit,len(ids),C):
            n=min(C,len(ids)-p)
            if policy=="native":skip=False
            elif policy=="regions":skip=reg.prepare([r],[p],[n])
            else:skip=ns.can_omit([r],[p],[n])
            omitted+=int(skip)*n
            if policy!="regions" or reg.may_publish(r,p+n):
                published=max(published,(p+n)//page*page)
        if published:q.append(ids[:published])
        row=dict(target=len(ids)-hit,draft=len(ids)-hit-omitted,omitted=omitted,hit=hit)
        records.append(row);stats.update(row)
    return dict(totals=dict(stats),requests=records)
def adversarial():
    series=array("q",range(74160))
    def other(n,j):return series[:1]+array("q",range(j,j+n-1))
    short,otherlong=other(97,150000),other(74160,446000)
    scenarios={"short_long":[(short,"a"),(series,"a")],
        "fork_and_return":[(series,"a"),(series[:32000]+short,"a"),(series+short,"a")],
        "short_same_bucket_other_salt":[(short,"b"),(series,"a"),(otherlong,"c")],
        "long_short_long_disjoint_bos":[(series,"a"),(short,"a"),(otherlong,"a")],
        "repeated_interior_branch":[(series,"a")]+[(series[:32768]+other(753,i*4123+11),"a") for i in range(3)]}
    return {str(p):{sc:{mode:experiments(tr,p,mode) for mode in ("native","sticky","promote","regions")} for sc,tr in scenarios.items()} for p in (1,64)}
def physical():
    out=Counter()
    for seed in range(30):
        gen=random.Random(seed)
        for page in (1,4):
            tree=ToyCache(window=8,page=page)
            cov=DFlashRegionCoverage(8,page,1+seed%3)
            adapter=S()
            adapter.may_publish=lambda r,n:cov.may_publish(with_salt(r),n)
            adapter.prepare=lambda rs,ps,es:cov.prepare([with_salt(r) for r in rs],ps,es)
            adapter.match_limit=lambda ids,maximum:cov.match_limit(request(ids,'institution'),ids,maximum)
            def with_salt(r):
                r.extra_key=None;r.cache_salt='institution'
                return r
            tree.cover=adapter
            base=list(range(96));tree.run(base)
            corpus=[base]
            for i in range(40):
                prior=gen.choice(corpus)
                br=prior[:gen.randint(1,len(prior))]+[gen.randrange(100,150) for _ in range(gen.randrange(1,35))]
                if i%7==0:tree.offload_reload()
                if i%11==0:tree.nodes.clear()
                if i%9==0:tree.run(br,abort_after=gen.randrange(1,len(br)+1))
                tree.run(br,[gen.randrange(100,150) for _ in range(gen.randrange(6))]);corpus.append(br)
            out.update(tree.stats)
    return dict(out)
def main():
    trials=adversarial();safety=physical()
    a=request(array("q",range(74160)),None);b=request(array("q",range(753)),None)
    c=DFlashRegionCoverage(W,1)
    assert not c.prepare([a,b],[16384,0],[16384,753]) and not c.dirty
    assert c.prepare([a,a],[16384,16384],[8192,8192])
    out=dict(page_source="startup_full_5183861a98ee4825.log server_args.page_size 1; 64 sensitivity",trials=trials,physical=safety,
        tiny_filter_bytes=[1,2,3],mixed_short_fallback=True,homogeneous_long_omission=True)
    (H/"ordering_stress.json").write_text(json.dumps(out,indent=2)+"\n")
    print(safety)
    for name,items in trials["1"].items():
        print(name,{mode:(v["totals"]["target"],v["totals"]["omitted"]) for mode,v in items.items()})
if __name__=="__main__":main()

