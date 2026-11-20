"""CPU-reviewed bridge to round4 configure/resolve API; no serving registration.

Only a B1 research execution is admitted. Mixed request grouping is deliberately
not implemented: width must never be selected from batch size/load. Worker source
still needs a same-stream post-draft decision handoff before target verify.
"""
from contextlib import contextmanager
import threading
from score_prefix_controller import agree_packets

class RequestPrefixBridge:
    def __init__(self,prefix_api,*,epoch,policy_signature):
        self.api=prefix_api;self.epoch=epoch;self.policy_signature=policy_signature
        self.thread=threading.get_ident();self.last_generation=0;self.active=False

    @contextmanager
    def scope(self,worker,batch,parallel,packets,*,world_size,request_key,generation):
        if threading.get_ident()!=self.thread or self.active:
            raise RuntimeError('foreign thread or nested request decision')
        if len(batch.reqs)!=1:
            raise RuntimeError('B1 research admission only; no batch width aggregation')
        if getattr(batch.reqs[0],'rid',None)!=request_key:
            raise RuntimeError('batch request does not own decision')
        if generation<=self.last_generation:
            raise RuntimeError('decision already consumed or stale')
        width=agree_packets(packets,world_size)
        p=packets[0]
        if (p.epoch!=self.epoch or p.generation!=generation or p.request_key!=request_key
                or p.policy_signature!=self.policy_signature):
            raise RuntimeError('decision is not for current request/model/policy')
        sentinel=object();prior=getattr(worker,'_prefix_verification_config',sentinel)
        self.active=True;self.last_generation=generation
        try:
            self.api.configure_prefix_verification(worker,width)
            resolved=self.api.resolve_prefix_verification_width(worker,batch,parallel)
            # Actual round4 guard may reject an unsupported topology. Make the
            # configuration itself agree with the default fallback, not just yield8.
            if resolved!=width:
                self.api.configure_prefix_verification(worker,resolved)
            yield resolved
        finally:
            if prior is sentinel:
                if hasattr(worker,'_prefix_verification_config'):
                    del worker._prefix_verification_config
            else: worker._prefix_verification_config=prior
            self.active=False

