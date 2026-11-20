"""Bounded request-owned suffix proposal index for eager CPU or CUDA PyTorch.

Only bind, truncate and release are host lifecycle operations. propose and
append_verified accept device-resident frontiers, epochs and per-row lengths.
The caller must order actions on a single stream per store, or supply CUDA
events between streams. Returned flags are tensors and cause no CPU readback.
Each hash bucket holds a recent end position. Direct-map collisions may miss
opportunities. Every key and source span is validated against live history.
"""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class RequestHandle:
    slot: int
    epoch: int
    request_id: str


class SuffixStore:
    suffixes = (8, 16)

    def __init__(self, slots, capacity, *, buckets=65536, device="cpu", stop_ids=()):
        if slots < 1 or capacity < 32 or buckets < 1 or buckets & (buckets - 1):
            raise ValueError(
                "need positive slots, capacity >=32 and power-of-two buckets"
            )
        self.slots, self.capacity, self.buckets = slots, capacity, buckets
        self.device = torch.device(device)
        self.stop_ids = tuple(int(x) for x in stop_ids)
        self.tokens = torch.empty(
            (slots, capacity + 1), device=device, dtype=torch.int32
        )
        self.ends = torch.full(
            (slots, 2, buckets + 1), -1, device=device, dtype=torch.int32
        )
        self.live = torch.zeros(slots, device=device, dtype=torch.int64)
        self.frontier = torch.zeros(slots, device=device, dtype=torch.int64)
        self.owners = [None] * slots
        self.serial = 0

    def _ids(self, x):
        return torch.as_tensor(x, device=self.device, dtype=torch.int64)

    def _windows(self, slot, end, k):
        col = end[..., None] + torch.arange(-k, 0, device=self.device)
        return self.tokens[slot[..., None], col.clamp(0, self.capacity - 1)].long()

    def _hash(self, key):
        # An odd-base rolling hash in int64. Low bits choose a power-of-two bucket.
        h = torch.zeros_like(key[..., 0])
        for part in key.unbind(-1):
            h = h * 1000003 + part + 1
        return torch.bitwise_and(h ^ (h >> 27), self.buckets - 1)

    def _publish(self, slot, end, valid):
        for i, k in enumerate(self.suffixes):
            key = self._windows(slot, end, k)
            h = self._hash(key)
            keep = valid & (end >= k)
            # Reserve a separate sink bucket to absorb padding and rejected rows.
            h = torch.where(keep, h, self.buckets)
            addr = slot * (2 * (self.buckets + 1)) + i * (self.buckets + 1) + h
            value = torch.where(keep, end, -1).to(torch.int32)
            self.ends.view(-1).scatter_reduce_(
                0, addr.reshape(-1), value.reshape(-1), reduce="amax", include_self=True
            )

    def bind(self, slot, request_id, confirmed_tokens):
        if not 0 <= slot < self.slots or self.owners[slot] is not None:
            raise ValueError("bind requires a free slot")
        ids = torch.as_tensor(confirmed_tokens, device=self.device)
        if (
            ids.ndim != 1
            or ids.numel() > self.capacity
            or ids.numel()
            and (
                ids.dtype not in (torch.int32, torch.int64)
                or bool((ids < 0).any())
                or (ids.dtype == torch.int64 and bool((ids >= 2**31).any()))
            )
        ):
            raise ValueError("invalid confirmed history")
        self.serial += 1
        handle = RequestHandle(slot, self.serial, str(request_id))
        self.live[slot] = self.serial
        self.frontier[slot] = ids.numel()
        self.ends[slot].fill_(-1)
        self.tokens[slot, : ids.numel()].copy_(ids)
        self._reindex(slot, ids.numel())
        self.owners[slot] = handle
        return handle

    def _reindex(self, slot, n):
        for start in range(0, n, 1024):
            end = torch.arange(start, min(n, start + 1024), device=self.device)
            self._publish(
                torch.full_like(end, slot), end, torch.ones_like(end, dtype=torch.bool)
            )

    def truncate(self, handle, expected_end, new_end):
        if self.owners[handle.slot] != handle or not 0 <= new_end <= expected_end:
            return False
        if self.frontier[handle.slot].item() != expected_end:
            return False
        self.frontier[handle.slot] = new_end
        self.ends[handle.slot].fill_(-1)
        self._reindex(handle.slot, new_end)
        # Invalidate stale writes even if a prior branch reaches this frontier.
        self.serial += 1
        renewed = RequestHandle(handle.slot, self.serial, handle.request_id)
        self.live[handle.slot] = renewed.epoch
        self.owners[handle.slot] = renewed
        return renewed

    def release(self, handle):
        if self.owners[handle.slot] != handle:
            return False
        self.live[handle.slot] = 0
        self.frontier[handle.slot] = 0
        self.owners[handle.slot] = None
        return True

    def _check(self, slots, epochs, expected):
        slot, epoch, n = map(self._ids, (slots, epochs, expected))
        if slot.ndim != 1 or epoch.shape != slot.shape or n.shape != slot.shape:
            raise ValueError("need aligned slot, epoch and frontier vectors")
        safe = slot.clamp(0, self.slots - 1)
        valid = (
            (slot == safe)
            & (epoch > 0)
            & (self.live[safe] == epoch)
            & (n == self.frontier[safe])
        )
        # Reject duplicate active mutations; an old-epoch request cannot poison
        # the current request occupying the same slot.
        same = (slot[:, None] == slot[None, :]) & (epoch[:, None] == epoch[None, :])
        return safe, n, valid & (same.sum(1) == 1)

    def append_verified(self, slots, epochs, expected, output, counts):
        # Output includes newly accepted tokens and the bonus, excludes the old anchor.
        slot, n, valid = self._check(slots, epochs, expected)
        out = self._ids(output)
        count = self._ids(counts)
        if out.ndim != 2 or out.shape[0] != slot.numel() or count.shape != slot.shape:
            raise ValueError("need a rectangular output and per-row counts")
        col = torch.arange(out.shape[1], device=self.device)
        used = col[None, :] < count[:, None]
        valid = (
            valid
            & (count >= 0)
            & (count <= out.shape[1])
            & (n + count <= self.capacity)
        )
        valid = valid & torch.all(~used | ((out >= 0) & (out < 2**31)), dim=1)
        end = n[:, None] + col
        keep = valid[:, None] & used
        address = slot[:, None] * (self.capacity + 1) + torch.where(
            keep, end, self.capacity
        )
        self.tokens.view(-1).scatter_(
            0, address.flatten(), out.to(torch.int32).flatten()
        )
        self._publish(slot[:, None].expand_as(end), end, keep)
        # amax prevents a stale or duplicate slot row from overwriting a new valid
        # frontier. Truncation is an explicit separate operation.
        self.frontier.scatter_reduce_(
            0, slot, torch.where(valid, n + count, 0), reduce="amax", include_self=True
        )
        return valid

    def propose(
        self, slots, epochs, expected, anchors, remaining, *, width=32, min_suffix=8
    ):
        if width < 2 or min_suffix not in self.suffixes:
            raise ValueError("unsupported proposal width or suffix threshold")
        slot, n, valid = self._check(slots, epochs, expected)
        anchors, remaining = map(self._ids, (anchors, remaining))
        if anchors.shape != n.shape or remaining.shape != n.shape:
            raise ValueError("need anchors and remaining output lengths")
        valid = valid & (n >= min_suffix) & (remaining >= width) & (n <= self.capacity)
        valid = valid & (
            self.tokens[slot, (n - 1).clamp(0, self.capacity - 1)] == anchors
        )
        source = torch.full_like(n, -1)
        for i, k in enumerate(self.suffixes):
            if k < min_suffix:
                continue
            current = self._windows(slot, n, k)
            end = self.ends[slot, i, self._hash(current)].long()
            other = self._windows(slot, end, k)
            exact = (
                valid & (n >= k) & (end >= k) & (end < n) & (other == current).all(-1)
            )
            for eos in self.stop_ids:
                exact = exact & (current != eos).all(-1)
            source = torch.where(exact, end, source)
        columns = torch.arange(width - 1, device=self.device)
        cand = self.tokens[
            slot[:, None], (source[:, None] + columns).clamp(0, self.capacity - 1)
        ].long()
        hit = valid & (source >= 0) & (source + width - 1 <= n)
        for eos in self.stop_ids:
            hit = hit & (cand != eos).all(dim=-1)
        proposed = torch.cat((anchors[:, None], cand), dim=1)
        return torch.where(hit[:, None], proposed, torch.zeros_like(proposed)), hit
