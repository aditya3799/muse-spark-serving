"""Conservative coverage for complete prefill-chunk omission on exclusively sliding DFlash.

An ordinary request may read any previously published prefix. A monotone bound on
possibly omitted logical positions survives canonical slot replacement and host offload.
"""
from array import array


class DraftWindowCoverage:
    def __init__(self, window, page_size, buckets=4096):
        self.window = int(window)
        self.key_length = max(1, int(page_size))
        self.high = array("Q", [0]) * buckets
        self.dirty = False

    def _bucket(self, ids):
        return hash(tuple(ids[:self.key_length])) % len(self.high)

    def prepare(self, reqs, prefixes, extends):
        """Mark before publication. A mixed batch with any required row runs stock."""
        if not reqs or prefixes is None or extends is None:
            return False
        if len(reqs) != len(prefixes) or len(reqs) != len(extends):
            return False
        batch = []
        for req, start, count in zip(reqs, prefixes, extends):
            if (req.session is not None or req.multimodal_inputs is not None
                or req.positional_embed_overrides is not None or req.skip_radix_cache_insert):
                return False
            tokens = req.full_untruncated_fill_ids
            end = start + count
            if count <= 0 or start < self.key_length or end > len(tokens) - self.window:
                return False
            batch.append((self._bucket(tokens),end))
        for index,end in batch:
            self.high[index] = max(self.high[index],end)
        self.dirty = True
        return True

    def unsafe_from(self, key_ids, length):
        if not self.dirty or length <= self.key_length or len(key_ids) < self.key_length:
            return None
        start = max(self.key_length,length-self.window)
        return start if start < min(length,self.high[self._bucket(key_ids)]) else None

    def match_limit(self, token_ids, limit):
        start = self.unsafe_from(token_ids,len(token_ids))
        if start is None: return limit
        return start if limit is None else min(limit,start)

    def may_publish(self, req, length):
        ids = req.origin_input_ids
        if len(ids) < self.key_length:
            ids = list(ids) + list(req.output_ids[:self.key_length-len(ids)])
        return self.unsafe_from(ids,length) is None

