"""Evaluate the frozen public 4-layer Qwen3 DFlash2 on observed prefixes, without a serving runtime."""
import ast
from pathlib import Path
import torch
import torch.nn.functional as F

here=Path(__file__).resolve().parent
ref=here/"dflash2_model_definitions.ref.txt"
conv_node=next(n for n in ast.parse(ref.read_text(encoding="utf-8")).body if getattr(n,"name",None)=="grouped_dynamic_conv")
scope={"torch":torch}
exec(compile(ast.Module([conv_node],[]),str(ref),"exec"),scope)
grouped=scope["grouped_dynamic_conv"]

def norm(x, weight):
    a=x.float()
    return (a*torch.rsqrt(a.square().mean(-1,keepdim=True)+1e-6)).to(x.dtype)*weight

class Drafter:
    def __init__(self,w,embed,head):
        self.w,self.embed,self.head=w,embed,head
    def projection(self,x,key):
        return F.linear(x,self.w[key+".weight"])
    def conv(self,x,key):
        w=self.w[key+".base_kernel"]
        h=self.projection(x,key+".kernel_projection").reshape(*x.shape[:-1],2,2,x.shape[-1]//16)
        return grouped(x,h[...,0,:,:],w[0],block_size=8,group_size=16),(h[...,1,:,:],w[1])
    def block_logits(self,taps,ids,anchors,compare_edges=False):
        device=taps.device
        n=len(anchors)
        width=8
        anchor=torch.as_tensor(anchors,device=device)
        # Restrict the tensor extent as well as the per-row mask to prefixes at/before the last anchor.
        length=max(anchors)
        cached=getattr(self,"prefix_kv",None)
        if cached is None:
            target=norm(self.projection(taps[:length],"fc"),self.w["hidden_norm.weight"])
        else:
            assert n==1 and self.prefix_length==length
        noise=torch.full((n,width),151669,dtype=torch.int64,device=device)
        noise[:,0]=ids[anchor]
        x=F.embedding(noise,self.embed)
        base_positions=torch.arange(length,device=device,dtype=torch.long)
        noise_positions=anchor[:,None]+torch.arange(width,device=device)[None,:]
        positions=torch.cat((base_positions.expand(n,-1),noise_positions),-1)
        inv=1./(1e6**(torch.arange(0,128,2,device=device).float()/128))
        angles=positions.float()[:,:,None]*inv[None,None]
        angle=torch.cat((angles,angles),-1)
        cos,sin=angle.cos()[:,None].to(x.dtype),angle.sin()[:,None].to(x.dtype)
        def rope(z,c,s):
            a,b=z.chunk(2,-1)
            return z*c+torch.cat((-b,a),-1)*s
        mask=torch.cat((base_positions[None,:]<anchor[:,None],torch.ones((n,width),device=device,dtype=torch.bool)),dim=-1)[:,None,None,:]
        assert length+width<=2048
        for layer in range(4):
            p=f"layers.{layer}."
            residual=x
            x,side=self.conv(norm(x,self.w[p+"input_layernorm.weight"]),p+"attention_conv")
            q=self.projection(x,p+"self_attn.q_proj").reshape(n,width,16,128)
            q=rope(norm(q,self.w[p+"self_attn.q_norm.weight"]).transpose(1,2),cos[:,:,-width:],sin[:,:,-width:])
            def kv(part):
                ctx=self.projection(target,p+"self_attn."+part+"_proj").expand(n,-1,-1)
                tail=self.projection(x,p+"self_attn."+part+"_proj")
                return torch.cat((ctx,tail),1).reshape(n,length+width,8,128)
            if cached is None:
                k=rope(norm(kv("k"),self.w[p+"self_attn.k_norm.weight"]).transpose(1,2),cos,sin)
                v=kv("v").transpose(1,2)
            else:
                kt=self.projection(x,p+"self_attn.k_proj").view(n,width,8,128)
                kt=rope(norm(kt,self.w[p+"self_attn.k_norm.weight"]).transpose(1,2),cos[:,:,-width:],sin[:,:,-width:])
                vt=self.projection(x,p+"self_attn.v_proj").view(n,width,8,128).transpose(1,2)
                cache_k,cache_v=cached[layer]
                k=torch.cat((cache_k[:length].transpose(0,1)[None],kt),2)
                v=torch.cat((cache_v[:length].transpose(0,1)[None],vt),2)
            attention=F.scaled_dot_product_attention(q,k.repeat_interleave(2,dim=1),v.repeat_interleave(2,dim=1),attn_mask=mask)
            output=self.projection(attention.transpose(1,2).reshape(n,width,-1),p+"self_attn.o_proj")
            x=residual+grouped(output,*side,block_size=width,group_size=16)
            residual=x
            x,side=self.conv(norm(x,self.w[p+"post_attention_layernorm.weight"]),p+"mlp_conv")
            up=self.projection(x,p+"mlp.up_proj")
            gate=self.projection(x,p+"mlp.gate_proj")
            output=self.projection(F.silu(gate)*up,p+"mlp.down_proj")
            x=residual+grouped(output,*side,block_size=width,group_size=16)
        hidden=norm(x,self.w["norm.weight"])[:,1:]
        logits=F.linear(hidden,self.head)
        unary,chosen=logits.topk(16,dim=-1)
        logmass=(unary.float().logsumexp(-1)-logits.float().logsumexp(-1))
        h=self.projection(hidden,"candidate_selector.hidden_projection")
        pred=torch.cat((ids[anchor,None,None].expand(-1,1,16),chosen[:,:-1]),dim=1)
        left=self.w["candidate_selector.predecessor_codebook"][pred]*h.unsqueeze(-2)
        right=self.w["candidate_selector.successor_codebook"][chosen]
        if getattr(self,"use_parent_edges",False) and not compare_edges:
            scores=torch.einsum("blpr,blcr->blpc",left,right)+unary.unsqueeze(-2)
            return chosen,scores,logmass
        scores=(left.unsqueeze(-2)*right.unsqueeze(-3)).sum(-1)+unary.unsqueeze(-2)
        if compare_edges:
            pinned_scores = torch.einsum("blpr,blcr->blpc",left,right)+unary.unsqueeze(-2)
            return chosen,scores,logmass,pinned_scores
        return chosen,scores,logmass
