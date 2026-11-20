"""Closed-loop CPU sensitivity replay of pinned Muse versus SSEngine prefill ordering."""
from __future__ import annotations

import ast
import json
import math
import random
import statistics
from pathlib import Path
from types import SimpleNamespace as S

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHUNK = 16_384
CAP = 1_893_608
BATCH_RATES = {1: 224.7, 4: 114.7, 16: 49.2, 32: 39.9}

def methods(path, class_name, names, namespace):
    original = ast.parse(path.read_text(encoding="utf-8-sig"))
    cls = next(n for n in original.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(cls.body) == len(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[class_name]

SG = methods(ROOT / "sglang/python/sglang/srt/managers/schedule_policy.py", "SchedulePolicy",
             {"_sort_by_shortest_prefill", "_shortest_prefill_work", "shortest_prefill_chunk_limit", "_uncached_len", "_sort_by_hrrn"},
             {"CacheAwarePolicy": S(SHORTEST_PREFILL_FIRST="spf"), "_ceil_div": lambda x,y: (x+y-1)//y})
SS = methods(ROOT / "research_2026-11-18_rl_experts/ssengine_reference/ssengine/engine/scheduler.py",
             "Scheduler", {"_form_srpt"}, {})
def lengths(rng, count, mean, span):
    if count == 1:
        return [mean]
    xs = [round(mean*(1-span + 2*span*i/(count-1))) for i in range(count)]
    xs[-1] += count*mean - sum(xs)
    rng.shuffle(xs)
    return xs

def percentile(xs, q):
    return sorted(xs)[math.ceil(q*len(xs))-1]

def decode_rate(n):
    if n in BATCH_RATES:
        return BATCH_RATES[n]
    lo = max(b for b in BATCH_RATES if b < n)
    hi = min(b for b in BATCH_RATES if b > n)
    return BATCH_RATES[lo] + (BATCH_RATES[hi]-BATCH_RATES[lo])*(n-lo)/(hi-lo)

class Replay:
    def __init__(self, c, seed, policy, stagger, cap):
        self.c, self.policy, self.cap = c, policy, cap
        self.now = self.tokens = self.active = self.preemptions = self.batches = 0
        self.wait, self.started, self.decode, self.events = [], [], [], []
        self.ttft = [[], []]
        self.age = float(policy[2:]) if policy.startswith("ss") else 0
        self.overdue = S(is_overdue=lambda r,t: t-r.arrival >= self.age) if self.age else None
        self.trace = []
        rng = random.Random(seed)
        context = [0]*c
        for turn in range(13):
            work = lengths(rng,c,74160 if turn == 0 else 753,.40)
            outs = lengths(rng,c,220,.65)
            session = []
            for i in range(c):
                context[i] += work[i]+outs[i]
                req = S(rid=f"{i:02d}:{turn:02d}", origin_input_ids=range(work[i]), output_ids=(),
                        num_matched_prefix_tokens=0, full_untruncated_fill_ids=range(work[i]), prefix_len=0,
                        out=outs[i]-1, work=work[i], remaining=work[i], turn=turn, i=i, footprint=context[i],
                        delay=rng.uniform(0,1.2 if turn == 0 else .30) if stagger else 0.)
                session.append(req)
            self.trace.append(session)
        self.events=[(r.delay,r) for r in self.trace[0]]
        self.sg = SG()
        self.sg.policy = policy
    def _schedule(self,r,n):
        self.alloc.append((r,n))
        r.remaining -= n
        r.prefix_len += n
    def eligible(self):
        return [r for r in self.wait if self.active+r.footprint<=self.cap]
    def admit(self,r):
        self.wait.remove(r)
        self.active+=r.footprint
    def _admit(self,budget,now_ns,no_longer_than=None,overdue_only=False):
        used=0
        while used<budget:
            ws=self.eligible()
            if overdue_only:
                ws=[r for r in ws if self.overdue.is_overdue(r,now_ns)]
                ws.sort(key=lambda r:r.arrival)
            else:
                ws.sort(key=lambda r:(r.work,r.arrival))
            if not ws or (no_longer_than is not None and ws[0].work>no_longer_than):
                break
            r=ws[0]
            self.admit(r)
            self._schedule(r,min(budget-used,r.remaining))
            used+=self.alloc[-1][1]
            if r.remaining:
                self.started.append(r)
        return used
    def prefill(self):
        self.alloc=[]
        prior=tuple(self.started)
        if self.policy.startswith("ss"):
            SS._form_srpt(self,CHUNK,self.now)
        else:
            if self.policy=="spf":
                SG._sort_by_shortest_prefill(self.wait,set())
            elif self.policy=="hrrn":
                SG._sort_by_hrrn(self.wait,set(),self.tokens)
            budget=CHUNK
            if self.started:
                r=self.started.pop()
                allowance=self.sg.shortest_prefill_chunk_limit(r,self.eligible(),budget,1) or budget
                self._schedule(r,min(r.remaining,allowance))
                budget-=self.alloc[-1][1]
                if r.remaining:
                    self.started.append(r)
            while budget:
                ws=self.eligible()
                if not ws:
                    break
                r=ws[0]
                self.admit(r)
                limit=min(budget,r.remaining)
                self._schedule(r,limit)
                budget-=limit
                if r.remaining:
                    assert not self.started
                    self.started.append(r)
        if not self.alloc:
            return False
        rows=sum(n for _,n in self.alloc)
        self.batches+=1
        self.tokens+=rows
        self.preemptions+=sum(1 for r in prior if r.remaining and all(a!=r for a,n in self.alloc))
        # Affine prefill cost gives C1=6.536s at 74,160 tokens in 5 chunks.
        # No attempt to fit official overlapping cold TTFT at other concurrency levels.
        self.now+=(6.536-5*.015)/74160*rows + .015
        for r,n in self.alloc:
            if not r.remaining:
                self.ttft[int(r.turn!=0)].append(self.now-r.arrival)
                self.decode.append(r)
        return True
    def run(self):
        while self.events or self.wait or self.started or self.decode:
            for ts,r in sorted(self.events.copy(),key=lambda v:(v[0],v[1].rid)):
                if ts>self.now+1e-10:
                    break
                self.events.remove((ts,r))
                r.arrival=r.arrival_ns=ts
                r.time_stats=S(wait_queue_entry_time=ts)
                r.arrival_processed_tokens=self.tokens
                self.wait.append(r)
            if self.prefill():
                continue
            if self.decode:
                self.now+=4/decode_rate(len(self.decode))
                for r in self.decode.copy():
                    r.out-=4
                    if r.out<=0:
                        self.active-=r.footprint
                        self.decode.remove(r)
                        if r.turn<12:
                            nxt=self.trace[r.turn+1][r.i]
                            self.events.append((self.now+nxt.delay,nxt))
                continue
            assert self.events and not self.started
            self.now=min(t for t,r in self.events)
        assert self.tokens==(74160+753*12)*self.c and self.active==0
        return dict(wall_s=self.now,first_p90=percentile(self.ttft[0],.90),first_p50=percentile(self.ttft[0],.50),
                    later_p90=percentile(self.ttft[1],.90),batches=self.batches,preempt=self.preemptions,tokens=self.tokens)

def main():
    raw=[]
    policies=("fcfs","spf","hrrn","ss0","ss5","ss30","ss60")
    for c in (1,4,16,32):
        for stagger in (False,True):
            for cap in ((CAP,10_000_000) if c==32 else (CAP,)):
                for seed in (7,19,53,97):
                    out={p:Replay(c,seed,p,stagger,cap).run() for p in policies}
                    raw.append(dict(c=c,stagger=stagger,cap=cap,seed=seed,out=out))
    stats={}
    for c in (1,4,16,32):
        primary=[x["out"] for x in raw if x["c"]==c and x["cap"]==CAP]
        stats[c]={}
        for p in policies:
            ratios=[x["fcfs"]["wall_s"]/x[p]["wall_s"] for x in primary]
            stats[c][p]=dict(min=min(ratios),median=statistics.median(ratios),max=max(ratios),
                              first_p90_ratio=statistics.median(x["fcfs"]["first_p90"]/x[p]["first_p90"] for x in primary),
                              later_p90_ratio=statistics.median(x["fcfs"]["later_p90"]/x[p]["later_p90"] for x in primary))
    (HERE/"replay.json").write_text(json.dumps(dict(summary=stats,scenarios=raw),indent=2))
    print(json.dumps(stats,indent=1))
if __name__=="__main__":
    main()


