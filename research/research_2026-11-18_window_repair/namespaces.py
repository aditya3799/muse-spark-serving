"""A bounded hint chooses a length floor; the explicit cache namespace supplies correctness."""
import json
from array import array


class DFlashWindowNamespaces:
    def __init__(self,window,page_size,buckets=4096):
        self.window=int(window)
        self.key_tokens=max(1,int(page_size))
        # Empty is -1. A collision may choose a smaller floor at the cost of extra work.
        self.floors=array("q",[-1])*buckets

    def assign(self,req):
        if hasattr(req,"_draft_window_floor"):
            return
        ids=req.origin_input_ids
        enabled=(req.session is None and req.multimodal_inputs is None
            and req.positional_embed_overrides is None and not req.skip_radix_cache_insert)
        length=len(ids)
        floor=0
        if enabled and length>=self.key_tokens:
            units=length//self.window
            proposed=0 if units<2 else self.window*(1<<(units.bit_length()-1))
            key=hash(tuple(ids[:self.key_tokens]))%len(self.floors)
            stored=self.floors[key]
            floor=proposed if stored<0 else min(proposed,stored)
            self.floors[key]=floor
        req._draft_window_floor=floor
        # All enabled-server traffic is tagged; an absent or caller-controlled raw
        # salt cannot alias another request's original salt or its length class.
        req.cache_salt=json.dumps(["dflash-window-v1",req.cache_salt,floor],separators=(",",":"))

    def can_omit(self,reqs,prefixes,extensions):
        if not reqs or prefixes is None or extensions is None:
            return False
        if len(reqs)!=len(prefixes) or len(reqs)!=len(extensions):
            return False
        for req,start,count in zip(reqs,prefixes,extensions):
            floor=getattr(req,"_draft_window_floor",0)
            if (count<=0 or floor>len(req.origin_input_ids)
                or start+count>floor-self.window):
                return False
        return True

