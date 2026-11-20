"""Select each live request's verification width from its own observed successes and the measured time curve."""
import numpy as np


def choose_widths(conditional, lengths, min_cost, max_cost, rounded=True):
    """Optimize expected confirmed tokens per unit time under linear packed-row cost.
    min_cost and max_cost include the common draft work; any graph rounding belongs in the cost table.
    """
    batch = len(lengths)
    extra_mass = np.cumprod(conditional, axis=1)
    valid = np.arange(1,8)[None,:] < np.asarray(lengths)[:,None]
    order = np.argsort(-np.where(valid,extra_mass,-1.).ravel(),kind="stable")[:valid.sum()]
    mass = extra_mass.ravel()[order]
    rows = np.arange(len(order)+1)
    gain = batch + np.r_[0.,mass.cumsum()]
    if rounded:
        catalogs=np.array([0,batch,3*batch,7*batch])
        rows=catalogs[np.searchsorted(catalogs,rows)]
    elapsed = min_cost + (max_cost-min_cost)*rows/(7*batch)
    take = np.argmax(gain/elapsed)
    return 1+np.bincount(order[:take]//7, minlength=batch)


class OnlineAcceptance:
    """Right-censored per-position Beta counts; each request belongs to its own current generation."""
    def __init__(self, count, strength=6):
        self.success=np.full((count,7), .75*strength)
        self.trials=np.full((count,7), float(strength))

    def predict(self, request):
        return self.success[request]/self.trials[request]

    def update(self, request, width, accepted):
        opportunities=np.arange(7)[None,:] < (width-1)[:,None]
        reached=np.arange(7)[None,:] <= accepted[:,None]
        successes=np.arange(7)[None,:] < accepted[:,None]
        self.trials[request] += opportunities & reached
        self.success[request] += opportunities & successes