"""Unintegrated certificate/lifetime contract for immutable captured weights.

The model owner MUST route reload through retire() before any .data/raw CUDA
write. Tensor versions cannot detect these writes. execution() must enclose each
host eager dispatch or graph.replay enqueue; a certificate is never baked into a
graph and assumed to remain valid without this host check. No backend enables it.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import threading
import weakref

def _identity(tensor):
    try:
        version=int(tensor._version)
        if version<0:return None
        return (int(tensor.data_ptr()),int(tensor.numel()),
                str(tensor.dtype),str(tensor.device),version)
    except (AttributeError,RuntimeError,TypeError,ValueError):
        return None

_LEASES=weakref.WeakKeyDictionary()
_LEASES_LOCK=threading.RLock()

class ImmutableWeightsLease:
    def __new__(cls,resident):
        # One lease per resident lifetime. In-place reload permanently disables it.
        # Only a new prepared resident object can receive a new certificate.
        with _LEASES_LOCK:
            existing=_LEASES.get(resident)
            if existing is not None:return existing
            self=super().__new__(cls)
            self.owner=weakref.ref(resident)
            self.lock=threading.RLock()
            self.live=True
            self.epoch=0
            self.active_enqueues=0
            _LEASES[resident]=self
            return self

    def __init__(self,resident):
        pass

    def retire(self,*,retire_graphs_and_synchronize):
        """Disable dispatch, retire all referencing graphs, wait for queued work.

        Only after this call returns may the owner mutate or replace weights.
        The callback must synchronize every model stream, not just the caller's
        current stream. A callback failure leaves the lease invalid.
        """
        if not callable(retire_graphs_and_synchronize):
            raise TypeError('explicit graph retirement/device synchronization required')
        with self.lock:
            if self.active_enqueues:
                raise RuntimeError('cannot retire inside an active enqueue scope')
            self.live=False
            self.epoch+=1
            retire_graphs_and_synchronize()

@dataclass(frozen=True)
class FiniteScaleProof:
    lease:object
    epoch:int
    stages:tuple
    identities:tuple

    def matches(self,resident):
        with self.lease.lock:
            if (not self.lease.live or self.lease.epoch!=self.epoch or
                    self.lease.owner() is not resident):return False
            for name,reference,identity in zip(('w13','w2'),self.stages,self.identities):
                tensor=getattr(resident,name,None)
                if reference() is not tensor or _identity(tensor)!=identity:return False
            return True

    @contextmanager
    def execution(self,resident):
        """Hold the owner's reload lock through eager/graph enqueue."""
        with self.lease.lock:
            if not self.matches(resident):
                raise RuntimeError('finite coefficient certificate is invalid')
            self.lease.active_enqueues+=1
            try:
                yield
            finally:
                self.lease.active_enqueues-=1

def scan_and_certify(resident,lease,scan_actual_resident):
    """Synchronously scan actual resident SF and bind both sides of the scan.

    scan_actual_resident is the verified packed-byte GPU scanner. It must wait
    for its result on the same device. No public boolean-to-proof shortcut.
    """
    with lease.lock:
        if not lease.live or lease.owner() is not resident:return None
        stages=tuple(getattr(resident,name,None) for name in ('w13','w2'))
        before=tuple(_identity(tensor) for tensor in stages)
        if any(x is None for x in before):return None
        epoch=lease.epoch
        finite=scan_actual_resident(resident)
        after=tuple(_identity(getattr(resident,name,None)) for name in ('w13','w2'))
        if (finite is not True or before!=after or not lease.live or lease.epoch!=epoch or
                any(getattr(resident,name,None) is not tensor for name,tensor in zip(('w13','w2'),stages))):
            return None
        return FiniteScaleProof(lease,epoch,tuple(weakref.ref(t) for t in stages),before)
