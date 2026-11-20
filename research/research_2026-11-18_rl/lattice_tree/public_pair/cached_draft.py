"""Use append-only, committed context K/V as the pinned DFlash serving path does."""
import torch
from draft import Drafter,norm

class CachedDrafter(Drafter):
    def begin_prefix(self,taps):
        h,d,c=8,128,2048
        if not hasattr(self,"prefix_kv"):
            self.prefix_kv=[tuple(torch.empty((c,h,d),device=taps.device,dtype=taps.dtype) for _ in range(2)) for _ in range(4)]
        self.prefix_length=0
        self.append_prefix(taps)
    def append_prefix(self,taps):
        start=self.prefix_length
        length=start+taps.shape[0]
        assert 0<=start<=length<=self.prefix_kv[0][0].shape[0]
        if start==length:return
        tar=norm(self.projection(taps,"fc"),self.w["hidden_norm.weight"])
        device=tar.device
        freq=1./(1e6**(torch.arange(0,128,2,device=device).float()/128))
        pos=torch.arange(start,length,device=device).float()[:,None]*freq
        angle=torch.cat((pos,pos),-1)
        c,s=angle.cos().to(tar.dtype)[:,None],angle.sin().to(tar.dtype)[:,None]
        for i,(keys,values) in enumerate(self.prefix_kv):
            p=f"layers.{i}.self_attn."
            key=norm(self.projection(tar,p+"k_proj").view(-1,8,128),self.w[p+"k_norm.weight"])
            a,b=key.chunk(2,-1)
            keys[start:length].copy_(key*c+torch.cat((-b,a),-1)*s)
            values[start:length].copy_(self.projection(tar,p+"v_proj").view(-1,8,128))
        self.prefix_length=length
