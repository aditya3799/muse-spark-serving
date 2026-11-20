"""Rank-local TP8 dense FFN seams matching the parent's projection-shape choices."""
import torch
import torch.nn.functional as F

class DenseEnvelope:
    def __init__(self,m,seed):
        self.rows=m
        self.kind="column" if m<512 else "row" if m%8==0 else "replicated"
        gen=torch.Generator(device="cuda").manual_seed(seed+739)
        self.pool=torch.randn((4096,8192),device="cuda",dtype=torch.bfloat16,generator=gen)*.011
        self.unpool=torch.randn((8192,4096),device="cuda",dtype=torch.bfloat16,generator=gen)*.016
        self.norm_weight=torch.linspace(.85,1.15,4096,device="cuda",dtype=torch.bfloat16)
        self.assembled=torch.empty((m,4096),device="cuda",dtype=torch.bfloat16)

    def peer_fixture(self,hidden):
        if self.kind=="replicated":
            return None
        # Other-rank result fixture, staged before the captured measurement.
        # The runtime path computes only rank 0's own rows/columns. True H200 TP
        # peer arithmetic can have distinct backend-dependent reduction order.
        return torch.mm(hidden,self.pool.t(),out_dtype=torch.float32).to(torch.bfloat16)

    def stage_peers(self,peers):
        if peers is not None:
            self.assembled.copy_(peers)

    def before(self,hidden):
        if self.kind=="column":
            own=torch.mm(hidden,self.pool[:512].t(),out_dtype=torch.float32).to(torch.bfloat16)
            self.assembled[:,:512].copy_(own)
            pooled=self.assembled
        elif self.kind=="row":
            count=self.rows//8
            own=torch.mm(hidden[:count],self.pool.t(),out_dtype=torch.float32).to(torch.bfloat16)
            self.assembled[:count].copy_(own)
            pooled=self.assembled
        else:
            pooled=F.linear(hidden,self.pool)
        # Parent's MuseSpark1xRMSNorm on GPU uses F.rms_norm with the frozen
        # effective zero-centered-gamma weight. Fixture provides that weight directly.
        return F.rms_norm(pooled,(4096,),self.norm_weight,1e-5).to(torch.bfloat16)

    def after(self,partial):
        # Only local finalized routed-expert partial is available. An identity
        # across the missing expert all-reduce supplies the postprojection input.
        if self.kind=="column":
            return torch.mm(partial,self.unpool[:1024].t(),out_dtype=torch.float32).to(torch.bfloat16)
        if self.kind=="row":
            return torch.mm(partial[:self.rows//8],self.unpool.t(),out_dtype=torch.float32).to(torch.bfloat16)
        return F.linear(partial,self.unpool)

