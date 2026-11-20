"""Install only the guarded prologue into the isolated native decoder fixture."""
import torch
import torch.nn.functional as F
from types import SimpleNamespace as NS
from prologue_package import Fa3Prologue

def install(op):
    def entry(self,hidden,pos,rotary,loc,table,new):
        packet=F.linear(hidden,self.wq)
        norm=NS(_head_dim=64,_num_q_heads=16,_num_kv_heads=2,_eps=1e-5)
        pa,ro=(None,None) if self.full else (pos,rotary)
        q=None
        if new:
            if not hasattr(self,'prologue_adapter'):self.prologue_adapter=Fa3Prologue()
            q=self.prologue_adapter.try_run(packet,pa,ro,self.k8,self.v8,loc,
                reference=lambda x,p,r:op.oldq(norm,x,p,r),reference_is_triton=True,compatible_pool=True)
        if q is None:
            q,k=op.oldq(norm,packet,pa,ro)
            op.store_cache(k.to(op.f8).view(torch.uint8),packet[:,1152:1280].to(op.f8).view(torch.uint8),
                self.k8.view(torch.uint8),self.v8.view(torch.uint8),loc,row_bytes=128,v_row_bytes=128,size_limit=self.k8.shape[0])
            q=q.to(op.f8)
        # Same source FP8 bytes expanded to BF16 for local SM86 native FA3.
        # This bridge is in every timed arm and is absent from Hopper serving.
        self.k.view(-1,128).index_copy_(0,loc,self.k8.view(torch.uint8)[loc].view(op.f8).to(op.b16))
        self.v.view(-1,128).index_copy_(0,loc,self.v8.view(torch.uint8)[loc].view(op.f8).to(op.b16))
        return q.to(op.b16).view(-1,16,64),packet[:,-1024:]
    op.Block.entry=entry
