import ast
import contextlib
import functools
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

here = Path(__file__).resolve().parent
root = here.parent
kernels = root / "research_2026-11-18_rl/lattice_tree"
sys.path.insert(0,str(kernels))
from fixed_tree_adapter import TreeStep, WIDTH

assert os.environ.get("TRITON_INTERPRET") == "1" and not torch.cuda.is_available()
torch.set_num_threads(1)


def native_worker_methods():
    source = root / "sglang/python/sglang/srt/speculative/dflash_worker_v2.py"
    src = source.read_text(encoding="utf-8-sig")
    parent = (root/"release_2026-11-18/candidate.diff").read_bytes()
    tree = ast.parse(src)
    workers = next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="DFlashWorkerV2")
    methods = [f for f in workers.body if isinstance(f,ast.FunctionDef)
               and f.name in {"_append_target_hidden_to_draft_kv_by_loc","_draft_write_ids"}]
    env={"torch":torch,"draft_tp_context":lambda _: contextlib.nullcontext()}
    for method in methods:
        exec(compile("from __future__ import annotations\n"+ast.unparse(method),str(source),"exec"),env)
    assert len(methods)==2
    return env, hashlib.sha256(parent).hexdigest()


class Pool:
    def __init__(self):
        self.k = torch.full((5,300,4), -999.,dtype=torch.float32)
        self.v = self.k.clone()

    def set_kv_buffer_prefix_valid(self,layer,slots,counts,key,value,*scales):
        for b,count in enumerate(counts.tolist()):
            where = slots[b,:count]
            self.k[layer.i,where]=key[b*8:b*8+count,0]
            self.v[layer.i,where]=value[b*8:b*8+count,0]


def worker(env,pool,events):
    def attn(i):
        return NS(attn=NS(i=i,k_scale=None,v_scale=None),num_kv_heads=1,head_dim=4,
                  kv_proj_only=lambda x:(x*(i+1),x-i),apply_k_norm=lambda x:x,
                  apply_k_rope=lambda pos,x:x+pos[:,None].float())
    model=NS(layers=[NS(self_attn=attn(i)) for i in range(5)],
             project_target_hidden=lambda x:x,prepare_context_hidden_for_kv=lambda layer,x:x)
    w=NS(model_runner=NS(device=torch.device("cpu"),shared_read_done_event=None),lilicorr=None,
         _need_mamba_verify_commit=False,_use_fused_kv_materialize=False,draft_parallel_layout=None,
         draft_model=model,draft_model_runner=NS(kv_index_translator="draft",token_to_kv_pool=pool))
    for s in ["_draft_write_ids","_append_target_hidden_to_draft_kv_by_loc"]:
        setattr(w,s,functools.partial(env[s],w))
    return w


def call_oracle(step,layer,raw_q,kc,vc,paths,physical,checking):
    # Independent serial single-path reference consumes canonical physical KV slots only.
    window=layer.sliding_window_size
    paths=[[int(physical[j]) for j in (p if window < 0 else p[-(window+1):])] for p in paths]
    def attention(q, pages):
        return torch.softmax((q*kc[pages]).sum(-1)*0.5,-1) @ vc[pages]
    wanted = torch.stack([attention(q,p) for q,p in zip(raw_q,paths)])
    def prep(q,qr,kr,layer,logical_batch_size,kv_head_num,is_prefill):
        assert logical_batch_size==len(paths) and kv_head_num==1 and is_prefill
        return q,None,None,torch.ones((len(paths),1)),torch.ones((len(paths),1))
    fa=NS(page_size=1,fa_impl_ver=3,use_mla=False,fa_skip_kv_cache=False,kv_cache_is_mxfp8=False,
          prepare_paged_mha_query=prep,get_paged_mha_kv_cache=lambda l:(kc,vc),num_splits=0)
    def oracle(**args):
        assert args["k_descale"].shape==(len(paths),1) and args["v_descale"].shape==(len(paths),1)
        assert args["cu_seqlens_q"].tolist()==list(range(len(paths)+1))
        assert args["cu_seqlens_k_new"].tolist()==[0]+args["cache_seqlens"].cumsum(0).tolist()
        assert args["max_seqlen_q"]==1 and args["causal"] and args["ver"]==3
        out=[]
        for r,length in enumerate(args["cache_seqlens"].tolist()):
            start=0 if window<0 else max(0,length-window-1)
            pages=args["page_table"][r,start:length].tolist()
            assert pages==paths[r]
            out.append(attention(args["q"][r,0],pages))
        checking["native_call_argument_and_serial_oracle_rows"]+=len(out)
        return torch.stack(out)
    out=step.attention(fa,layer,raw_q,oracle)
    assert torch.equal(out,wanted)
    return out


def case(data,prompt,anchor_indices,env,receipt):
    b=len(anchor_indices)
    prefix=torch.tensor([11,17,9,13],dtype=torch.int32)[:b]
    req=torch.tensor([4,1,5,2])[:b]
    table=torch.zeros((8,48),dtype=torch.int32)
    gen=torch.Generator().manual_seed(100*prompt+anchor_indices[0]+17)
    # Different request namespaces intentionally share five immutable prefix pages.
    ids=torch.randperm(210,generator=gen,dtype=torch.int32)+1
    used=5
    for r,l in zip(req,prefix):
        table[r,:5]=ids[:5]
        table[r,5:l+8]=ids[used:used+l+3]
        used+=int(l)+3
    original=table.clone()
    owned=torch.stack([table[r,l:l+8].clone() for r,l in zip(req,prefix)])
    perm=lambda:torch.randperm(256,generator=gen,dtype=torch.int32)
    full,swa,draft=perm(),perm(),perm()
    step=TreeStep(b+1,32,"cpu",window=4).active(b)
    loc=data["anchors"][anchor_indices]
    anchors=torch.from_numpy(data["tokens"][loc]).long()
    id_tensor=torch.from_numpy(data["ids"][anchor_indices]).long()
    scores=torch.from_numpy(data["scores"][anchor_indices])
    step.propose(id_tensor,scores,anchors)
    step.stage(table,req,prefix,owned,full,swa,4)
    path_nodes=[]
    virtual=[]
    for i in range(b):
        for n in range(WIDTH):
            nodes=[n]
            while nodes[-1]!=0:
                nodes.append(int(step.parents[i,nodes[-1]]))
            nodes=nodes[::-1]
            assert int(step.positions[i*8+n])==int(prefix[i])+len(nodes)-1
            assert int(step.mask[i,n])==sum(1<<x for x in nodes)
            path_nodes.append(nodes)
            virtual.append(original[req[i],:prefix[i]].tolist()+owned[i,nodes].tolist())
    q=torch.randn((b*8,4),generator=gen)
    h=q.clone()
    for window,mapping in [(-1,full),(4,swa)]:
        layer=NS(sliding_window_size=window,is_cross_attention=False,tp_q_head_num=1,tp_k_head_num=1,
                 head_dim=4,scaling=.5,logit_cap=0)
        kc,vc=[torch.randn((300,4),generator=gen) for _ in range(2)]
        h+=call_oracle(step,layer,h,kc,vc,virtual,mapping,receipt)
    # Only a proposal on the actual AR prefix has an observed target in the public trace.
    # Off-path top1 is assigned a distinct sentinel; those rows must never be consumed.
    target=torch.full_like(step.tokens,-123456)
    expected=[]
    for i,idx in enumerate(loc):
        gold=data["tokens"][idx+1:idx+9]
        seen=[0]
        for j in range(8):
            parent=seen[-1]
            target[i,parent]=int(gold[j])
            child=[n for n in range(1,8) if int(step.parents[i,n])==parent and int(step.tokens[i,n])==int(gold[j])]
            if not child:break
            assert len(child)==1
            seen+=child
        expected.append(seen)
        receipt["real_qwen_checked_tokens"]+=len(seen)
        receipt["nonordinal_accepted_paths"]+=int(seen!=list(range(len(seen))))
        receipt["branching_proposals"]+=int(any(int(step.parents[i,n])!=n-1 for n in range(1,8)))
    step.commit(target,table,req,prefix,owned)
    pool=Pool(); events=[]
    w=worker(env,pool,events)
    physical_draft=draft[owned.long()].long()
    plan=NS(write_ids=lambda translator:physical_draft.reshape(-1) if translator=="draft" else None)
    ev=NS(record=lambda:events.append("late WAR"))
    def published(new_lens):
        assert w.model_runner.shared_read_done_event is ev
        events.append("publish")
    step.append_and_publish(w,plan,h,published,ev)
    assert events==["late WAR","publish"]
    wanted_table=original.clone()
    for i,seen in enumerate(expected):
        n=len(seen); r=req[i];l=prefix[i]
        rel=seen+[k for k in range(8) if k not in seen]
        wanted_table[r,l:l+8]=owned[i,rel]
        assert step.src_at_dest[i].tolist()==rel
        assert int(step.counts[i])==n and int(step.new_lens[i])==l+n
        gold=data["tokens"][loc[i]+1:loc[i]+n+1].tolist()
        assert step.emitted[i,:n].tolist()==gold and int(step.bonus[i])==gold[-1]
        for layer in range(5):
            for rank,node in enumerate(seen):
                page=int(draft[int(table[r,l+rank])])
                assert torch.equal(pool.k[layer,page],h[i*8+node]*(layer+1)+float(l+rank))
                assert torch.equal(pool.v[layer,page],h[i*8+node]-layer)
                receipt["accepted_layer_kv_pairs"]+=1
        # Next contiguous reservation begins at new_len. A write into a rejected
        # physical page cannot alter any previously committed prefix or another request.
        assert not set(owned[i,rel[:n]].tolist()) & set(owned[i,rel[n:]].tolist())
    assert torch.equal(table,wanted_table)
    outside=torch.ones(300,dtype=torch.bool); outside[physical_draft.reshape(-1)]=False
    assert (pool.k[:,outside]==-999).all() and (pool.v[:,outside]==-999).all()
    receipt["requests"]+=b


def main():
    env,parent=native_worker_methods()
    r=dict(requests=0,branching_proposals=0,nonordinal_accepted_paths=0,real_qwen_checked_tokens=0,
           native_call_argument_and_serial_oracle_rows=0,accepted_layer_kv_pairs=0)
    for prompt in [2,5,8,11]:
        d=np.load(kernels/"public_pair"/f"trace_{prompt:02}.npz")
        for first in [0,12,24,36,48,60]:
            case(d,prompt,list(range(first,first+4)),env,r)
    result=dict(scope="CPU Triton interpreter, held-out Qwen conditional lattices and AR outputs; pinned worker method bodies, reduced synthetic KV and capture of native FA3 call arguments; native FA3 kernel is not executed.",parent_sha256=parent,**r)
    print(json.dumps(result,indent=2))
    (here/"fixed_tree_adapter_cpu.json").write_text(json.dumps(result,indent=2))
if __name__=="__main__": main()

