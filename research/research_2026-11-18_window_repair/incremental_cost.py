"""Bounded incremental cryptographic prefix state with content-checked mutation repair."""
from array import array
from collections import OrderedDict
import hashlib,json
from pathlib import Path
from regions import DFlashRegionCoverage
import scheduler_cost,stress
class IncrementalCoverage(DFlashRegionCoverage):
    def __init__(self,*args,cache_budget=16<<20,**kw):
        super().__init__(*args,**kw)
        self.index=OrderedDict();self.cache_budget=cache_budget;self.cached=0
    def _indexes(self,ids,begin,namespace):
        if not isinstance(ids,array) or ids.typecode!="q":ids=array("q",ids[:begin])
        # Keep only immutable snapshots, keyed by identity as a hint. A mutated
        # or recycled object must pass native byte equality before digest reuse.
        raw=memoryview(ids)[:begin].cast("B").tobytes()
        key=(id(ids),namespace);entry=self.index.pop(key,None)
        if entry is not None:self.cached-=len(entry[0])
        valid=entry is not None and (entry[0].startswith(raw) if len(raw)<=len(entry[0]) else raw.startswith(entry[0]))
        if valid:
            prior,h,digests=entry
        else:
            h=hashlib.blake2b(digest_size=16,person=b"dflash-region-v2")
            h.update(json.dumps(namespace,separators=(",",":")).encode()+b"\0")
            prior=b"";digests=[h.digest()]
        block=self.region*8
        for i in range(len(prior),len(raw),block):
            h.update(memoryview(raw)[i:i+block]);digests.append(h.copy().digest())
        if len(raw)>len(prior):prior=raw
        a,b=__import__("struct").unpack("<QQ",digests[begin//self.region])
        if len(prior)<=self.cache_budget:
            self.index[key]=(prior,h,digests);self.cached+=len(prior)
            while len(self.index)>64 or self.cached>self.cache_budget:
                _,old=self.index.popitem(last=False);self.cached-=len(old[0])
        n=8*len(self.bits);b|=1
        return ((a+j*b)%n for j in range(3))
def independent_keys(c):
    x=array("q",range(8192))
    y=array("q",x)
    space=("x","s")
    def key(obj,n):return tuple(c._indexes(obj,n,space))
    def fresh(obj,n):return tuple(IncrementalCoverage(2048,1)._indexes(obj,n,space))
    for n in [2048,6144,2048,4096,8192]:assert key(x,n)==fresh(x,n)==key(y,n)
    x[3000]+=11
    for n in [4096,2048,8192,4096,6144]:assert key(x,n)==fresh(x,n)
    del x[6050:];x.extend(range(3333))
    for n in [8192,2048,4096]:assert key(x,n)==fresh(x,n)
    for n in range(78):key(array("q",[n])*8192,8192)
    assert len(c.index)<=64 and c.cached<=c.cache_budget
if __name__=="__main__":
    for budget in (0,128,2048*8,1048576):independent_keys(IncrementalCoverage(2048,1,cache_budget=budget))
    scheduler_cost.DFlashRegionCoverage=IncrementalCoverage;stress.DFlashRegionCoverage=IncrementalCoverage
    result={str(b):scheduler_cost.bench(b,16384//b,rounds=9) for b in (1,16)}
    result["physical"]=stress.physical()
    result["stable_after_suffix_edits_retracts_identity_and_lru_eviction"]=True
    H=Path(__file__).resolve().parent;(H/"incremental_cost.json").write_text(json.dumps(result,indent=2))
    print({k:v["median"] for k,v in result.items() if k in ("1","16")},result["physical"])

