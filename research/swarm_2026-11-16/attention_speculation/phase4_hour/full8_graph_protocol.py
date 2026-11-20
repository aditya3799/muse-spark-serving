"""Full8 graph/accept-prefix control contract, default off.
DeviceScalar below is a CPU stand-in for one capture-stable int32 CUDA scalar.
A real mask kernel must LOAD that scalar during replay; Python specialization
on its value would be wrong. This module is not installed in SGLang.
"""
from prefix_protocol import accept_prefix

class DeviceScalar:
    def __init__(self,value):self.value=value
    def load(self):return self.value

class Fixed8PrefixController:
    def __init__(self,enabled=False):
        self.enabled=enabled;self.device_width=DeviceScalar(8)
        self.active=False;self.pending_aux=False;self.generation=0
    def set_width(self,width):
        if self.active:raise RuntimeError('Cannot change K until acceptance and aux commit finish')
        if width not in (1,2,4,6,8):raise ValueError('Unsupported prefix width')
        self.device_width.value=width if self.enabled else 8
    def begin(self,candidates,positions,cache_slots):
        if self.active:raise RuntimeError('One target replay at a time')
        if any(len(row)!=8 for group in (candidates,positions,cache_slots) for row in group):raise ValueError('Target graph shape must remain8')
        if not(len(candidates)==len(positions)==len(cache_slots)):raise ValueError('Batch mismatch')
        self.width=self.device_width.load();self.active=True;self.pending_aux=True;self.generation+=1;self.accepted=False
        self.candidates=candidates;self.positions=positions;self.cache_slots=cache_slots
        # Existing target graph and host bound stay8, regardless of acceptance K.
        return dict(draft_token_num=8,num_tokens_per_req=8,output_stride=8,
                    tokens=candidates,positions=positions,cache_slots=cache_slots,
                    mask_control=self.device_width,generation=self.generation)
    def accept(self,target_top1,prefix_lengths):
        if not self.active:raise RuntimeError('No live target replay')
        if self.accepted:raise RuntimeError('Duplicate acceptance for this target generation')
        bs=len(self.candidates)
        if len(target_top1)!=bs or len(prefix_lengths)!=bs:raise ValueError('Exact target/prefix batch lengths required')
        if any(len(row)!=8 for row in target_top1):raise ValueError('Expected full8 target output')
        # Validate/compute locally, then publish atomically. Failed acceptance
        # cannot leave partial commits visible to the materializer.
        commits=[accept_prefix(c[:self.width],p[:self.width],n) for c,p,n in zip(self.candidates,target_top1,prefix_lengths)]
        self.commits=commits;self.accepted=True
        return self.commits
    def materialize(self,hidden,generation):
        if not self.active or not self.pending_aux or generation!=self.generation:raise RuntimeError('Stale or missing auxiliary output')
        if not self.accepted:raise RuntimeError('Successful acceptance must precede materialization')
        if len(hidden)!=len(self.candidates) or any(len(row)!=8 for row in hidden):raise ValueError('Retain full8 FC/materializer input shape')
        # Full8 projection receives every row; stock prefix-valid writer persists
        # only input rows[:accepted+1]. Bonus KV remains pending, never tail aux.
        committed=[row[:c.committed] for row,c in zip(hidden,self.commits)]
        self.pending_aux=False
        return dict(full_hidden=hidden,full_positions=self.positions,full_cache_slots=self.cache_slots,
                    commit_lengths=[c.committed for c in self.commits],written_hidden=committed)
    def end(self):
        if not self.active or self.pending_aux:raise RuntimeError('Auxiliary output must be consumed before next target')
        self.active=False
        if hasattr(self,'commits'):del self.commits
