"""Balance entire expert slots from earlier route counts under a fixed migration budget."""
import numpy as np

RANKS = 8
SLOTS = 32
EXPERTS = RANKS * SLOTS
NATIVE = np.arange(EXPERTS, dtype=np.int32).reshape(RANKS, SLOTS)
EXPERT_BYTES = 3 * 4096**2 * (1/2 + 1/16) + 2*2


def work(counts, tile):
    counts = np.asarray(counts, dtype=np.int64)
    return (counts + tile - 1) // tile


def select(counts, swap_limit, *, tile=16):
    """Exchange slots between two ranks, leaving each with precisely 32 unique experts."""
    cost = work(counts, tile)
    slots = NATIVE.copy()
    plan = []
    for _ in range(swap_limit):
        rank_load = cost[slots].sum(axis=1)
        best = None
        for src in np.flatnonzero(rank_load == rank_load.max()):
            for dst in np.flatnonzero(rank_load < rank_load[src]):
                move = cost[slots[src, :, None]] - cost[slots[dst, None, :]]
                # Reduce the squared load imbalance. An exchange transfers move tiles.
                change = 2 * move * (rank_load[src] - rank_load[dst] - move)
                i,j = np.unravel_index(np.argmax(change), change.shape)
                merit = int(change[i,j])
                if merit > 0 and (best is None or merit > best[0]):
                    best = merit, int(src), int(dst), int(i), int(j)
        if best is None:
            break
        _,src,dst,i,j = best
        slots[src,i], slots[dst,j] = slots[dst,j], slots[src,i]
        plan.append((src,i,dst,j))
    assert np.array_equal(np.sort(slots.ravel()), np.arange(EXPERTS))
    return slots,plan


def route_counts(ids):
    return np.bincount(ids.ravel(), minlength=EXPERTS)


def local_map(slots, rank):
    mapping = np.full(EXPERTS, -1, dtype=np.int32)
    mapping[slots[rank]] = np.arange(SLOTS,dtype=np.int32)
    return mapping


def peak(counts, slots, tile):
    return int(work(counts,tile)[slots].sum(axis=1).max())


def demand(seed, pattern):
    rng = np.random.default_rng(seed)
    if pattern == "normal":
        return .03 * rng.standard_normal(EXPERTS) / np.sqrt(-np.log(np.clip(rng.random(EXPERTS),1e-12,1)))
    if pattern == "skewed":
        return 4 * rng.random(EXPERTS)**6
    return np.zeros(EXPERTS)


def draw(seed, count, heat):
    rng=np.random.default_rng(seed)
    scores=rng.random((count,EXPERTS),dtype=np.float32)+heat
    return np.argpartition(scores,-8,axis=1)[:,-8:].astype(np.int32)
