"""CPU oracle only. Same-run counts, immutable plans, no GPU or serving imports."""
from dataclasses import dataclass
import math, uuid

@dataclass(frozen=True)
class Plan:
    run_id:str
    epoch:int
    holders:tuple
    slots:tuple
    counts:tuple

class Budget:
    def __init__(self,ranks,limit=2_000_000_000):
        self.limit=limit;self.live=[0]*ranks;self.peak=[0]*ranks;self.reservations={}
    def reserve(self,key,rank,resident,temporary=0):
        if key in self.reservations:raise ValueError('duplicate reservation')
        if min(resident,temporary)<0 or self.live[rank]+resident+temporary>self.limit:raise ValueError('per-GPU peak budget')
        self.peak[rank]=max(self.peak[rank],self.live[rank]+resident+temporary)
        self.live[rank]+=resident;self.reservations[key]=(rank,resident)
    def release(self,key):
        rank,n=self.reservations.pop(key);self.live[rank]-=n

class Session:
    def __init__(self,experts=256,ranks=8):
        if experts%ranks:raise ValueError('EP division')
        self.experts=experts;self.ranks=ranks;self.local=experts//ranks
        self.run_id=uuid.uuid4().hex;self.epoch=0;self.counts=[0]*experts
    def observe(self,rows):
        delta=[0]*self.experts
        for row in rows:
            live=[e for e in row if e!=-1]
            if len(set(live))!=len(live):raise ValueError('top-k duplicates')
            for e in live:
                if type(e)is not int or not 0<=e<self.experts:raise ValueError('expert ID')
                delta[e]+=1
        self.counts=[a+b for a,b in zip(self.counts,delta)]
        self.epoch+=1
    def baseline(self):
        h=tuple((e//self.local,) for e in range(self.experts))
        return self._plan(h)
    def _plan(self,holders):
        # Existing local slots stay untouched. Replicas append to reserved capacity.
        slots=[[None]*self.experts for _ in range(self.ranks)]
        for e in range(self.experts):slots[e//self.local][e]=e%self.local
        for r in range(self.ranks):
            n=self.local
            for e,hs in enumerate(holders):
                if r in hs and r!=e//self.local:slots[r][e]=n;n+=1
        return Plan(self.run_id,self.epoch,tuple(tuple(h)for h in holders),tuple(tuple(s)for s in slots),tuple(self.counts))
    def choose(self,budget,max_copies=8,copy_bytes=28_311_552,temporary_bytes=28_311_552,layer='layer'):
        holders=[list(h)for h in self.baseline().holders]
        def predicted(hs):
            loads=[0.0]*self.ranks
            for e,owners in enumerate(hs):
                for r in owners:loads[r]+=self.counts[e]/len(owners)
            return tuple(sorted(loads,reverse=True))
        for step in range(max_copies):
            old=predicted(holders);best=None
            for e in range(self.experts):
                if not self.counts[e]:continue
                for r in range(self.ranks):
                    if r in holders[e] or budget.live[r]+copy_bytes+temporary_bytes>budget.limit:continue
                    trial=[v[:] for v in holders];trial[e].append(r);trial[e].sort()
                    score=predicted(trial)
                    if score<old and (best is None or (score,e,r)<best[:3]):best=(score,e,r,trial)
            if best is None:break
            _,e,r,holders=best
            budget.reserve((layer,self.run_id,self.epoch,e,r),r,copy_bytes,temporary_bytes)
        return self._plan(holders)
    def route(self,rows,plan,policy='ordinal'):
        if plan.run_id!=self.run_id:raise ValueError('cross-run plan')
        if plan.epoch>self.epoch:raise ValueError('future plan')
        counts=[0]*self.experts;out=[[] for _ in range(self.ranks)]
        for token,row in enumerate(rows):
            live=[e for e in row if e!=-1]
            if len(set(live))!=len(live):raise ValueError('top-k duplicates')
            rankrows=[[] for _ in range(self.ranks)]
            for e in row:
                if e==-1:
                    for v in rankrows:v.append(-1)
                    continue
                if type(e)is not int or not 0<=e<self.experts:raise ValueError('expert ID')
                hs=plan.holders[e]
                if policy not in ('ordinal','token_mod','whole'):raise ValueError('policy')
                owner=hs[(counts[e] if policy=='ordinal' else token)%len(hs)] if policy!='whole' else hs[0]
                counts[e]+=1
                for r,v in enumerate(rankrows):v.append(plan.slots[r][e] if r==owner else -1)
            for r in range(self.ranks):out[r].append(rankrows[r])
        return out

def analyze(rows,plan,rank_ids,bm=16):
    rank_counts=[[0]*len(plan.holders)for _ in rank_ids]
    for t,row in enumerate(rows):
        for k,e in enumerate(row):
            owners=[r for r in range(len(rank_ids)) if rank_ids[r][t][k]>=0]
            assert len(owners)==(e!=-1),'exactly-once'
            if e!=-1:
                r=owners[0];assert r in plan.holders[e]
                assert rank_ids[r][t][k]==plan.slots[r][e]
                rank_counts[r][e]+=1
    # Counts/tile work are conditional proxies, not latency predictions.
    routes=[sum(c)for c in rank_counts]
    tiles=[sum((n+bm-1)//bm for n in c)for c in rank_counts]
    return dict(rank_routes=routes,max_rank_routes=max(routes),rank_m_tiles=tiles,max_rank_m_tiles=max(tiles),total_m_tiles=sum(tiles),scope='route-count and fixed-BM tile proxies only, no timing')
