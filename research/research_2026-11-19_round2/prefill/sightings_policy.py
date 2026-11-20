"""Pure-host bounded admission policy; not a distributed production adapter.

All ranks must feed the identical ordinary-TP shape sequence and apply the same
collective qualification result. Published plans are never evicted because
captured graphs can retain their workspace/descriptor pointers.
"""
from collections import OrderedDict

class SightingsPolicy:
    def __init__(self,min_sightings=8,metadata_capacity=64,plan_capacity=8):
        assert min_sightings>=2 and metadata_capacity>=plan_capacity>=1
        self.minimum=min_sightings;self.capacity=metadata_capacity;self.plan_capacity=plan_capacity
        self.seen=OrderedDict();self.pending=set();self.active=set();self.rejected=set()
    def visit(self,shape):
        if shape in self.active:return 'fast'
        if shape in self.rejected:return 'native_rejected'
        if shape in self.pending:return 'validate'
        if len(self.active)+len(self.pending)>=self.plan_capacity:return 'native_capacity'
        count=self.seen.pop(shape,0)+1;self.seen[shape]=count
        while len(self.seen)>self.capacity:self.seen.popitem(last=False)
        return 'screen' if count>=self.minimum else 'native_unseen'
    def outcome(self,shape,phase,accepted):
        assert phase in ('screen','validate')
        if not accepted:
            self.pending.discard(shape);self.rejected.add(shape);self.seen.pop(shape,None)
        elif phase=='screen':
            self.pending.add(shape);self.seen.pop(shape,None)
        else:
            assert shape in self.pending
            self.pending.remove(shape);self.active.add(shape)

