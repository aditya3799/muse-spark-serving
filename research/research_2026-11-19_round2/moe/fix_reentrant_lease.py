from pathlib import Path
here=Path(__file__).resolve().parent/'finite_lifetime_v2';p=here/'finite_scale_state.py';s=p.read_text(encoding='utf-8-sig')
s=s.replace('class ImmutableWeightsLease:\n    def __init__(self,resident):\n        self.owner=weakref.ref(resident)\n        self.lock=threading.RLock()\n        self.live=True\n        self.epoch=0', '''_LEASES=weakref.WeakKeyDictionary()
_LEASES_LOCK=threading.RLock()

class ImmutableWeightsLease:
    def __new__(cls,resident):
        # One live lease per resident. A new generation requires prior retirement.
        with _LEASES_LOCK:
            existing=_LEASES.get(resident)
            if existing is not None and existing.live:return existing
            self=super().__new__(cls)
            self.owner=weakref.ref(resident)
            self.lock=threading.RLock()
            self.live=True
            self.epoch=0
            self.active_enqueues=0
            _LEASES[resident]=self
            return self

    def __init__(self,resident):
        pass''')
s=s.replace('with self.lock:\n            self.live=False','with self.lock:\n            if self.active_enqueues:\n                raise RuntimeError(\'cannot retire inside an active enqueue scope\')\n            self.live=False')
s=s.replace("                raise RuntimeError('finite coefficient certificate is invalid')\n            yield", "                raise RuntimeError('finite coefficient certificate is invalid')\n            self.lease.active_enqueues+=1\n            try:\n                yield\n            finally:\n                self.lease.active_enqueues-=1")
p.write_text(s,encoding='utf-8')
p=here/'test_state.py';s=p.read_text(encoding='utf-8-sig');needle='report=dict(complete=True'
add='''x,lease,p=setup();assert ImmutableWeightsLease(x) is lease;cases.append('canonical_live_lease')
with p.execution(x):
 try:lease.retire(retire_graphs_and_synchronize=lambda:None);raise AssertionError('reentrant retirement accepted')
 except RuntimeError:pass
 assert p.matches(x)
assert lease.active_enqueues==0;cases.append('reentrant_retirement_rejected')
lease.retire(retire_graphs_and_synchronize=lambda:None)
newlease=ImmutableWeightsLease(x);assert newlease is not lease and not p.matches(x);cases.append('new_generation_requires_retirement')
'''
s=s.replace(needle,add+needle);p.write_text(s,encoding='utf-8')
