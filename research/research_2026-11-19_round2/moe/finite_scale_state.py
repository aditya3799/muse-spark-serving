"""Layer-local finite-SF evidence. No Torch import and no runtime activation.

A captured graph requires immutable weights for its whole lifetime. This class
checks eager/pre-capture identity/version changes; it cannot inspect Python state
when a captured graph is replayed. Reload must retire/rebuild graphs.
"""
from dataclasses import dataclass
import weakref

def _identity(tensor):
    try:
        version=int(tensor._version)
        if version<0:return None
        return (id(tensor),int(tensor.data_ptr()),int(tensor.numel()),
                str(tensor.dtype),str(tensor.device),version)
    except (AttributeError,RuntimeError,TypeError,ValueError):
        return None

@dataclass(frozen=True)
class FiniteScaleProof:
    owner:object
    stages:tuple
    verified_finite:bool

    @classmethod
    def from_verified_scan(cls,resident,finite):
        # finite must be the result of scanning both ACTUAL source scale tensors.
        # API separation keeps the CPU metadata code free of GPU imports.
        identities=tuple(_identity(getattr(resident,s)) for s in ('w13','w2'))
        return cls(weakref.ref(resident),identities,
                   finite is True and all(x is not None for x in identities))

    def matches(self,resident):
        if not self.verified_finite or self.owner() is not resident:return False
        return self.stages==tuple(_identity(getattr(resident,s,None)) for s in ('w13','w2'))
