"""Request-local proposal contract. Isolated CPU research, not installed in serving.

History includes the already target-confirmed anchor exactly once. KV prefix
ends immediately before that anchor. Lookup supplies proposals only; acceptance
always uses target predictions. No shared corpus, external tokens or load input.
"""
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class Candidate:
    request_key: str
    epoch: int
    version: int
    context_length: int
    source_end: int
    match_length: int
    tokens: Tuple[int, ...]  # anchor + proposals

@dataclass(frozen=True)
class Commit:
    accepted_drafts: int
    emitted: Tuple[int, ...]
    cached_inputs: Tuple[int, ...]
    bonus: int

class RequestLookup:
    def __init__(self, request_key, context, *, min_match=4, max_match=16,
                 max_rows=32, occurrences_per_key=64):
        if not request_key or not (1 <= min_match <= max_match):
            raise ValueError('invalid request/match configuration')
        if max_rows < 2 or occurrences_per_key < 1:
            raise ValueError('invalid capacity')
        self.request_key = request_key
        self.min_match, self.max_match = min_match, max_match
        self.max_rows, self.occurrences_per_key = max_rows, occurrences_per_key
        self.epoch, self.version = 0, 0
        self._closed = False
        self._reset(context)

    @staticmethod
    def _validate_tokens(tokens):
        values = tuple(tokens)
        if any(type(t) is not int or t < 0 for t in values):
            raise ValueError('token IDs must be nonnegative integers')
        return values

    def _reset(self, context):
        values = self._validate_tokens(context)
        if not values:
            raise ValueError('context must include confirmed anchor')
        self._history = list(values)
        self._index = defaultdict(lambda: deque(maxlen=self.occurrences_per_key))
        self._index_new_ends(self.min_match)

    def _index_new_ends(self, first_end):
        h = self._history
        for end in range(max(first_end, self.min_match), len(h)+1):
            self._index[tuple(h[end-self.min_match:end])].append(end)

    @property
    def context(self):
        return tuple(self._history)

    def close(self):
        self._history.clear(); self._index.clear()
        self.epoch += 1; self._closed = True

    def replace_context(self, context):
        if self._closed:
            raise ValueError('closed request')
        self._reset(context)
        self.epoch += 1; self.version = 0

    def propose(self, *, remaining_output, eos_ids=()):
        if self._closed:
            raise ValueError('closed request')
        if type(remaining_output) is not int or remaining_output < 1:
            return None
        h = self._history; n = len(h)
        if n < self.min_match:
            return None
        # At most W outputs including bonus; do not reserve beyond output cap.
        limit = min(self.max_rows-1, remaining_output-1)
        if limit < 1:
            return None
        stop = set(eos_ids)
        best = None; best_key = None
        for end in reversed(self._index.get(tuple(h[-self.min_match:]), ())):
            if end >= n:
                continue
            match = self.min_match
            while match < min(self.max_match, end, n) and h[end-match-1] == h[n-match-1]:
                match += 1
            proposal = h[end:min(n,end+limit)]
            # Keep EOS for target bonus handling; no draft tokens after EOS.
            for i, token in enumerate(proposal):
                if token in stop:
                    proposal = proposal[:i]; break
            if not proposal:
                continue
            key = (match, len(proposal), end)
            if best_key is None or key > best_key:
                best_key = key
                best = Candidate(self.request_key, self.epoch, self.version,
                                 n, end, match, (h[-1], *proposal))
        return best

    def validate(self, candidate):
        if self._closed or candidate.request_key != self.request_key:
            raise ValueError('foreign or closed request')
        if (candidate.epoch, candidate.version, candidate.context_length) != (self.epoch,self.version,len(self._history)):
            raise ValueError('stale candidate')
        h=self._history; e=candidate.source_end; m=candidate.match_length
        if not (self.min_match <= m <= self.max_match and m <= e < len(h)):
            raise ValueError('invalid provenance')
        if not (2 <= len(candidate.tokens) <= self.max_rows):
            raise ValueError('invalid width')
        if h[e-m:e] != h[-m:] or candidate.tokens[0] != h[-1]:
            raise ValueError('invalid match or anchor')
        if tuple(h[e:e+len(candidate.tokens)-1]) != candidate.tokens[1:]:
            raise ValueError('proposal is not from this request')

    def accept_target_predictions(self, candidate, target_predictions):
        """Oracle contract, not a GPU/server acceptance implementation."""
        self.validate(candidate)
        pred=self._validate_tokens(target_predictions)
        if len(pred) != len(candidate.tokens):
            raise ValueError('target prediction width mismatch')
        k=0
        while k < len(pred)-1 and candidate.tokens[k+1] == pred[k]:
            k+=1
        bonus=pred[k]
        emitted=candidate.tokens[1:k+1]+(bonus,)
        cached=candidate.tokens[:k+1]
        old=len(self._history)
        self._history.extend(emitted)
        self._index_new_ends(old+1)
        self.version+=1
        return Commit(k, emitted, cached, bonus)
