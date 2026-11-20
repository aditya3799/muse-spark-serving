"""CPU oracle for branch reuse, prefix ownership, duplicate-index rebinding and eviction."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
from types import SimpleNamespace
from coverage_policy import DraftWindowCoverage


class ToyCache:
    def __init__(self,lookup_guard=True,publish_guard=True,window=8,page=4,chunk=16,buckets=4):
        self.cover=DraftWindowCoverage(window,page,buckets)
        self.lookup_guard,self.publish_guard=lookup_guard,publish_guard
        self.page,self.chunk,self.window=page,chunk,window
        self.nodes={}
        self.next_slot=0
        self.stats=Counter()

    def alloc(self,valid):
        self.next_slot+=1
        return self.next_slot,valid

    def query(self,tokens):
        maximum=len(tokens)-1
        if self.lookup_guard:
            maximum=self.cover.match_limit(tokens,maximum)
        maximum=maximum//self.page*self.page
        n=0
        while n<maximum and tuple(tokens[:n+1]) in self.nodes: n+=1
        n=n//self.page*self.page
        return [self.nodes[tuple(tokens[:i+1])] for i in range(n)]

    def checkpoint(self,req,tokens,loc):
        if self.publish_guard and not self.cover.may_publish(req,len(loc)):
            self.stats["withheld_publications"]+=1
            return
        lim=len(loc)//self.page*self.page
        for i in range(lim):
            key=tuple(tokens[:i+1])
            loc[i]=self.nodes.setdefault(key,loc[i])

    def run(self,tokens,generation=(),abort_after=None):
        req=SimpleNamespace(origin_input_ids=tokens,output_ids=[],full_untruncated_fill_ids=tokens,
            session=None,multimodal_inputs=None,positional_embed_overrides=None,skip_radix_cache_insert=False)
        slots=self.query(tokens)
        pos=len(slots)
        self.stats["reused_target_rows"]+=pos
        while pos<len(tokens):
            count=min(self.chunk,len(tokens)-pos)
            skip=self.cover.prepare([req],[pos],[count])
            self.stats["skipped_rows"]+=count*int(skip)
            slots.extend(self.alloc(not skip) for _ in range(count))
            pos+=count
            self.checkpoint(req,tokens,slots)
            if abort_after is not None and pos>=abort_after:
                return
        sequence=list(tokens)
        for tok in [None,*generation]:
            if tok is not None:
                req.output_ids.append(tok)
                sequence.append(tok)
                slots.append(self.alloc(True))
                self.checkpoint(req,sequence,slots)
            self.stats["reads"]+=min(self.window,len(slots))
            if any(not v for _,v in slots[-self.window:]):
                raise AssertionError("draft would consume stale canonical KV")

    def offload_reload(self):
        # The tree restores the same sidecar contents to different physical slots.
        self.nodes={k:self.alloc(v[1]) for k,v in self.nodes.items()}


def run():
    original=list(range(80))
    branch=original[:40]+[1000]
    for lookup,publish in ((False,True),(True,False),(True,True)):
        tree=ToyCache(lookup,publish)
        tree.run(original)
        tree.offload_reload()
        failed=False
        try:tree.run(branch)
        except AssertionError:failed=True
        assert failed==(not(lookup and publish)),(lookup,publish)
    stats=Counter()
    for seed in range(80):
        gen=random.Random(seed)
        cache=ToyCache(buckets=1+(seed%4))
        base=list(range(79+seed%27))
        cache.run(base,[100,101,102])
        corpus=[base]
        for j in range(35):
            on=gen.choice(corpus)
            cut=gen.randrange(1,len(on)+1)
            branch=on[:cut]+[gen.randrange(1001,1050) for _ in range(gen.randrange(1,39))]
            if j%7==0: cache.offload_reload()
            if j%11==0: cache.nodes.clear()
            if j%9==0: cache.run(branch,abort_after=gen.randrange(1,len(branch)+1))
            cache.run(branch,[gen.randrange(55,98) for _ in range(gen.randrange(0,9))])
            corpus.append(branch)
        stats.update(cache.stats)
    # A mixed batch with one participant close to decode must remain untouched.
    cov=DraftWindowCoverage(8,4)
    def req(n):
        return SimpleNamespace(session=None,multimodal_inputs=None,positional_embed_overrides=None,
            skip_radix_cache_insert=False,full_untruncated_fill_ids=list(range(n)))
    assert not cov.prepare([req(80),req(20)],[16,16],[16,4]) and not cov.dirty
    assert cov.prepare([req(80),req(70)],[16,16],[16,16])
    data=dict(random_runs=80,branch_requests=2800,one_guard_only_failed=True,raw=dict(stats),
              mixed_fallback_pass=True,scope="logical and physical-slot CPU oracle, every live read checked")
    out=Path(__file__).resolve().with_name("safety.json")
    out.write_text(json.dumps(data,indent=2)+"\n")
    print(data)


if __name__=="__main__":run()

