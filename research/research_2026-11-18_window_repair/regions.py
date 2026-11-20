"""Bounded logical KV coverage for ordinary local-cache requests."""
from array import array
import hashlib
import json


class DFlashRegionCoverage:
    def __init__(self,window,page_size,bytes_limit=1<<20):
        self.window=int(window)
        self.region=max(self.window,int(page_size))
        self.bits=bytearray(bytes_limit)
        self.dirty=False

    def _indexes(self,ids,begin,namespace):
        # All keys before the first complete region stay native. The logical
        # prefix through begin must agree for any matching slot in that region.
        if not isinstance(ids,array):
            ids=array("q",ids[:begin])
        h=hashlib.blake2b(digest_size=16)
        h.update(json.dumps(namespace,separators=(",",":")).encode())
        h.update(b"\0")
        h.update(begin.to_bytes(8,"little"))
        h.update(memoryview(ids)[:begin].cast("B"))
        raw=h.digest()
        a,b=int.from_bytes(raw[:8],"little"),int.from_bytes(raw[8:],"little")|1
        n=8*len(self.bits)
        return ((a+j*b)%n for j in range(3))

    def _mark(self,ids,start,end,namespace):
        for b in range(start//self.region,(end-1)//self.region+1):
            for ix in self._indexes(ids,b*self.region,namespace):
                self.bits[ix//8]|=1<<(ix%8)
        self.dirty=True

    def prepare(self,reqs,prefixes,extensions):
        if not reqs or prefixes is None or extensions is None or len(reqs)!=len(prefixes) or len(reqs)!=len(extensions):
            return False
        pending=[]
        for r,start,n in zip(reqs,prefixes,extensions):
            tokens=r.full_untruncated_fill_ids
            if (r.session is not None or r.multimodal_inputs is not None or r.positional_embed_overrides is not None
                or r.skip_radix_cache_insert or start<self.region or n<=0 or start+n>len(tokens)-self.window):
                return False
            pending.append((tokens,start,start+n,(r.extra_key,r.cache_salt)))
        for one in pending:self._mark(*one)
        return True

    def unsafe_from(self,ids,length,namespace):
        if not self.dirty or length<=self.region:
            return None
        first=max(self.region,length-self.window)
        for b in range(first//self.region,(length-1)//self.region+1):
            begin=b*self.region
            if all(self.bits[ix//8]&(1<<(ix%8)) for ix in self._indexes(ids,begin,namespace)):
                return max(first,begin)
        return None

    def match_limit(self,req,ids,limit):
        start=self.unsafe_from(ids,len(ids),(req.extra_key,req.cache_salt))
        return limit if start is None else start if limit is None else min(limit,start)

    def may_publish(self,req,length):
        if not self.dirty:return True
        ids=req.full_untruncated_fill_ids
        if len(ids)<length:
            ids=req.origin_input_ids+req.output_ids
        return self.unsafe_from(ids,length,(req.extra_key,req.cache_salt)) is None

