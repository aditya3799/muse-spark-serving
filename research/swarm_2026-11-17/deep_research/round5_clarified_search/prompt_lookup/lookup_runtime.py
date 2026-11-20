"""Default-off fixed-W8 lookup injection contract; CPU checks only so far.

All-hit skips backbone/sampler, but preserves stock embedding, compact-cache
rebuild, translator binding, target verification and target-hidden KV append.
Mixed-hit still runs the full draft batch, replacing only own-request hit rows.
No TP8 installer: a real collective-consistent payload protocol is required.
"""
from dataclasses import dataclass
from typing import Tuple
import torch
from request_lookup import RequestLookup

@dataclass(frozen=True)
class Snapshot:
    request_id: str
    request_epoch: int
    tokens: Tuple[int,...]  # includes most recent target-confirmed anchor
    remaining_output: int
    eos_ids: Tuple[int,...]=()

class Plan:
    def __init__(self, rows, mask):
        if rows.ndim != 2 or rows.shape[1] != 7 or mask.shape != (rows.shape[0],):
            raise ValueError('fixed8 plan shape mismatch')
        if rows.dtype != torch.int64 or mask.dtype != torch.bool or rows.device != mask.device:
            raise ValueError('plan dtype/device mismatch')
        self.rows, self.mask = rows, mask
        self.all_hit=bool(mask.all().item()) # explicit control sync; created before draft
    def merge(self, original):
        if original.shape != self.rows.shape or original.device != self.rows.device:
            raise ValueError('draft output mismatch')
        return torch.where(self.mask[:,None],self.rows,original)

class Fixed8LookupProposer:
    def __init__(self, snapshot_provider, *, enabled=False, world_size=1):
        if world_size != 1:
            raise ValueError('TP>1 requires qualified rank-consensus payload protocol')
        self.enabled=enabled;self.snapshot_provider=snapshot_provider
        self._states={}

    def forget(self, request_id):
        old=self._states.pop(request_id,None)
        if old:old[1].close()

    def __call__(self, *, worker, batch, anchor_tokens, prefix_lens):
        if not self.enabled:
            return None
        sampling=batch.sampling_info
        if (worker.block_size != 8 or sampling is None or not sampling.is_all_greedy
            or batch.has_grammar or batch.return_logprob
            or getattr(worker,'_is_domino',False) or getattr(worker,'lilicorr',None) is not None):
            return None
        snapshots=tuple(self.snapshot_provider(batch))
        if len(snapshots)!=int(anchor_tokens.numel()) or len({s.request_id for s in snapshots})!=len(snapshots):
            raise ValueError('snapshot batch identity mismatch')
        # Visible, accounted sync. Never read stale req.output_ids as current.
        reqs=tuple(batch.reqs)
        if len(reqs)!=len(snapshots) or any(snap.request_id != req.rid for snap,req in zip(snapshots,reqs)):
            raise ValueError('snapshot order does not match actual request IDs')
        # The harness must attach an epoch tied to each actual request lifecycle.
        if any(getattr(req,'_lookup_request_epoch',None) != snap.request_epoch for snap,req in zip(snapshots,reqs)):
            raise ValueError('request lifecycle epoch missing or stale')
        anchors=anchor_tokens.detach().cpu().tolist()
        prefixes=prefix_lens.detach().cpu().tolist()
        rows=[];hits=[]
        for snap,anchor,prefix in zip(snapshots,anchors,prefixes):
            tokens=tuple(snap.tokens)
            if not tokens or tokens[-1]!=anchor or len(tokens)!=prefix+1:
                rows.append([0]*7);hits.append(False);continue
            prior=self._states.get(snap.request_id)
            if prior is None or prior[0]!=snap.request_epoch:
                state=RequestLookup(snap.request_id,tokens,max_rows=8)
                self._states[snap.request_id]=(snap.request_epoch,state)
            else:
                state=prior[1]
                # Conservative rebuild until incremental scheduler wiring is qualified.
                if state.context!=tokens:state.replace_context(tokens)
            c=state.propose(remaining_output=snap.remaining_output,eos_ids=snap.eos_ids)
            hit=c is not None and len(c.tokens)==8
            if hit:state.validate(c)
            rows.append(list(c.tokens[1:]) if hit else [0]*7);hits.append(hit)
        if not any(hits):return None
        return Plan(torch.tensor(rows,dtype=torch.int64,device=anchor_tokens.device),
                    torch.tensor(hits,dtype=torch.bool,device=anchor_tokens.device))
