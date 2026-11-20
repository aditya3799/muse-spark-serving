"""CPU model of bounded two-phase replica publication; not a CUDA protocol."""
import hashlib,json
class Ledger:
    def __init__(self,ranks,limit=2_000_000_000):
        self.limit=limit;self.live=[0]*ranks;self.peak=[0]*ranks;self.items={}
    def allocate(self,key,rank,n):
        if key in self.items or n<0 or self.live[rank]+n>self.limit:raise ValueError('allocation budget/key')
        self.items[key]=(rank,n);self.live[rank]+=n;self.peak[rank]=max(self.peak[rank],self.live[rank])
    def free(self,key):
        rank,n=self.items.pop(key);self.live[rank]-=n

def validate(plan,ranks,experts,capacities):
    if len(plan.holders)!=experts or len(plan.slots)!=ranks:raise ValueError('metadata dimensions')
    for e,holders in enumerate(plan.holders):
        if not holders or len(set(holders))!=len(holders):raise ValueError('empty/duplicate holders')
        for rank in holders:
            if type(rank)is not int or not 0<=rank<ranks:raise ValueError('rank bounds')
            slot=plan.slots[rank][e]
            if type(slot)is not int or not 0<=slot<capacities[rank]:raise ValueError('slot capacity')
    for rank in range(ranks):
        slots=[plan.slots[rank][e]for e,h in enumerate(plan.holders)if rank in h]
        if len(slots)!=len(set(slots)):raise ValueError('expert slot alias')
    return hashlib.sha256(json.dumps([plan.run_id,plan.epoch,plan.holders,plan.slots],separators=(',',':')).encode()).hexdigest()

class Publication:
    def __init__(self,initial,capacities):
        self.active=initial;self.capacities=tuple(capacities);self.ranks=len(capacities)
        self.leases=[0]*self.ranks;self.pending=None;self.ready={};self.expected=None;self.poisoned=False;self.copy_started=False
    def enter(self,rank):
        if self.poisoned:raise RuntimeError('aborted copy may have overwritten active weights; rebuild required')
        if self.pending is not None:raise RuntimeError('quiescent publication in progress')
        self.leases[rank]+=1;return self.active
    def leave(self,rank):
        if self.leases[rank]<1:raise RuntimeError('lease underflow')
        self.leases[rank]-=1
    def prepare(self,plan,source_hashes):
        if self.poisoned:raise RuntimeError('poisoned publication')
        if self.pending is not None or any(self.leases):raise RuntimeError('old graph still live or pending update')
        if plan.run_id!=self.active.run_id or plan.epoch<self.active.epoch:raise ValueError('run/epoch')
        digest=validate(plan,self.ranks,len(plan.holders),self.capacities)
        self.pending=plan;self.expected=(digest,dict(source_hashes));self.ready={};self.copy_started=False
        return digest
    def begin_copy(self):
        if self.pending is None:raise RuntimeError("copy outside pending generation")
        self.copy_started=True
    def mark_ready(self,rank,digest,copied_hashes):
        if self.pending is None or digest!=self.expected[0]:raise ValueError('different metadata generation')
        required={e:self.expected[1][e]for e,h in enumerate(self.pending.holders)if rank in h}
        if copied_hashes!=required:raise ValueError('wrong/missing expert bytes or scale metadata')
        self.copy_started=True;self.ready[rank]=digest
    def publish(self):
        if self.pending is None or set(self.ready)!=set(range(self.ranks)):raise RuntimeError('incomplete ranks')
        if any(self.leases):raise RuntimeError('old producer/consumer live')
        self.active=self.pending;self.pending=None;self.ready={};self.expected=None
        return self.active
    def abort(self):
        # Quiescing alone cannot undo bytes already overwritten in reused slots.
        if self.copy_started:self.poisoned=True
        self.pending=None;self.ready={};self.expected=None
