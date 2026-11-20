"""Run live tree and chain DFlash under one real verifier, including causal masking, cache commit and planning."""
import json
import statistics
import sys
import time
from pathlib import Path
import numpy as np
import torch
here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parent))
from selector_tree import build_tree,walk

def trim(cache,previous,keep,linear,n):
    if not linear and keep!=list(range(len(keep))):
        index=torch.tensor(keep,device="cuda",dtype=torch.long)
        for layer in cache.layers:
            for buf in (layer.keys,layer.values):
                fresh=buf[:,:,previous:previous+n,:]
                buf[:,:,previous:previous+len(keep),:].copy_(fresh.index_select(2,index))
    cache.crop(previous+len(keep))
    assert cache.get_seq_length()==previous+len(keep)
NATIVE = None

def propose(candidates,scores,mass,anchor,policy):
    if NATIVE is not None:
        tree,chain = NATIVE
        if policy == "chain":
            return chain.chain(candidates,scores,anchor)
        return tree.single(candidates,scores,anchor,policy,mass)
    device=scores.device
    if policy=="chain":
        prev=torch.zeros(1,dtype=torch.long,device=device)
        chain=[anchor]
        for j in range(7):
            prev=scores[0,j].index_select(0,prev).argmax(-1)
            chain.append(candidates[0,j].index_select(0,prev))
        return torch.cat(chain),list(range(-1,7)),torch.arange(8,device=device),None
    n,temp,m,fork=policy.split("_")
    covered=mass.exp()[:,:,None].expand(-1,-1,16) if int(m[1:]) else None
    nodes,parent,depth,mask=build_tree(candidates,scores.float()/float(temp[1:]),anchor,
        int(n[1:]),covered,after_fork=float(fork[1:]))
    return nodes[0],parent[0].tolist(),depth[0],mask[0]
def taps(out):
    return torch.cat([out.hidden_states[k+1][0] for k in (2,10,18,26)],-1)
def measured(model,drafter,prompt,gold,policy):
    prefix=torch.tensor(prompt,device="cuda").long()
    pref=model(input_ids=prefix[None],use_cache=True,output_hidden_states=True,logits_to_keep=1)
    aux=taps(pref)
    if hasattr(drafter,"begin_prefix"):drafter.begin_prefix(aux)
    anchor=pref.logits[0,-1].argmax().reshape(1)
    cache=pref.past_key_values
    del pref
    # The first token from prompt prefill is already produced under both arms.
    assert int(anchor)==gold[0]
    produced=[int(anchor)]
    steps=[];wrong=0;inspection=None;inspection_cost=0.
    guarded=policy.endswith("_guard")
    base_policy=policy.removesuffix("_guard")
    fallback_calls=0
    torch.cuda.synchronize()
    total_start=time.perf_counter()
    while len(produced)<len(gold):
        start=time.perf_counter();previous=prefix.numel()
        ids=torch.cat((prefix,anchor))
        cand,scores,mass=drafter.block_logits(aux,ids,[previous])
        nodes,p,depth,mask=propose(cand,scores,mass,anchor,base_policy)
        def verify(nodes,p,depth,mask):
            n=nodes.numel()
            entry={"past_key_values":cache,"input_ids":nodes[None],"use_cache":True,"output_hidden_states":True,"logits_to_keep":0}
            if mask is not None:
                keep=torch.cat((torch.ones(n,previous,device="cuda",dtype=torch.bool),mask),1)
                bias=torch.zeros_like(keep,dtype=torch.bfloat16).masked_fill_(~keep,torch.finfo(torch.bfloat16).min)
                entry.update(attention_mask=bias[None,None],position_ids=(previous+depth)[None])
            target=model(**entry)
            next_ids=target.logits[0].argmax(-1).tolist()
            emitted,retained=walk(nodes.tolist(),p,next_ids,n+1)
            return target,emitted,retained
        target,emitted,retained=verify(nodes,p,depth,mask)
        linear=base_policy=="chain"
        if guarded:
            values=target.logits[0,retained].topk(2,dim=-1).values
            # Route borderline target decisions through the original eight-row chain.
            high,low=values[:,0].contiguous(),values[:,1]
            rounded=high.bfloat16()
            ulp=torch.nextafter(rounded,torch.full_like(rounded,float("inf"))).float()-rounded.float()
            if bool(((high.float()-low.float())<=2*ulp).any()):
                del target
                cache.crop(previous)
                nodes,p,depth,mask=propose(cand,scores,mass,anchor,"chain")
                target,emitted,retained=verify(nodes,p,depth,mask)
                fallback_calls+=1;linear=True
        n=nodes.numel()
        assert len(emitted)==len(retained)
        expected=gold[len(produced):len(produced)+len(emitted)]
        if inspection is None and getattr(drafter,"inspect_first_error",False):
            bad=next((j for j,(a,b) in enumerate(zip(expected,emitted)) if a!=b),None)
            if bad is not None:
                from audit_greedy import collect
                t0=time.perf_counter()
                inspection=collect(prefix,nodes,retained,bad,target.logits,expected[bad])
                torch.cuda.synchronize()
                inspection_cost+=time.perf_counter()-t0
        wrong+=sum(a!=b for a,b in zip(expected,emitted))
        produced.extend(emitted[:len(expected)])
        selected=torch.tensor(retained,device="cuda")
        trim(cache,previous,retained,linear,n)
        accepted_taps=taps(target).index_select(0,selected)
        if hasattr(drafter,"append_prefix"):drafter.append_prefix(accepted_taps)
        aux=torch.cat((aux,accepted_taps))
        prefix=torch.cat((prefix,nodes.index_select(0,selected)))
        anchor=torch.tensor(emitted[-1:],device="cuda")
        del target
        torch.cuda.synchronize()
        steps.append(dict(seconds=time.perf_counter()-start,verified=n,committed=len(expected),depth=len(retained)))
    first_mismatch=next((i for i,(a,b) in enumerate(zip(produced,gold)) if a!=b),None)
    elapsed=time.perf_counter()-total_start-inspection_cost
    if inspection is not None:
        from audit_greedy import replay
        inspection=replay(model,inspection)
    return dict(seconds=elapsed,checked=len(produced),wrong=wrong,steps=steps,first_mismatch=first_mismatch,produced=produced,first_error_analysis=inspection,fallback_verifications=fallback_calls)
def run_live(model,drafter,policies):
    summary=dict(scope="Public Qwen3 surrogate, one RTX 3050, one request at a time. Timings include DFlash forward, tree/chain planner, native HF target forward, decision and KV/tap commit. Prompt prefill excluded. Both paths use the same real weights. This Python implementation is not the pinned Muse deployment.",comparisons=[])
    import importlib
    import draft as public_draft
    fresh=importlib.reload(public_draft)
    drafter.block_logits=fresh.Drafter.block_logits.__get__(drafter)
    from check_future import shape_checks
    try:
        shape_checks(model,drafter)
        summary["masked_future_check"]="passed"
    except (AssertionError,RuntimeError) as error:
        # The live singleton reruns the drafter with genuinely present context only.
        # Preserve its independent output even if the separate batched-trace probe fails.
        summary["masked_future_check"]=repr(error)
        print("teacher-forced fidelity",repr(error),flush=True)
    drafter.use_parent_edges=True
    summary["edge_formula"]="Both live arms use the scored SGLang model's eager einsum edge expression. Teacher-forced traces used the upstream public checkpoint author's product/sum expression."
    global NATIVE
    try:
        from validate_planner import validated
        tree,chain,checked=validated()
        NATIVE=(tree,chain)
        summary["planner"]={"compiled":True,"checked_random_BF16_cases":checked}
    except Exception as err:
        summary["planner"]={"compiled":False,"reason":repr(err)}
    print("live planner",summary["planner"],flush=True)
    warm=np.load(here/"trace_00.npz")
    start=int(warm["start"]);tokens=warm["tokens"].tolist()
    for arm in ("chain",policies[8]["policy"]):
        measured(model,drafter,tokens[:start],tokens[start:start+9],arm)
    for i,p in enumerate(sorted(here.glob("trace_*.npz"))):
        idx=int(p.stem.split("_")[-1])
        if idx%2==0:continue
        trace=np.load(p);start=int(trace["start"]);tokens=trace["tokens"].tolist()
        gold=tokens[start:start+40]
        if len(gold)<9:continue
        arms=["chain",policies[8]["policy"]]
        if (idx//2)%2:arms.reverse()
        row={"id":idx,"arms":{}}
        for arm in arms:row["arms"][arm]=measured(model,drafter,tokens[:start],gold,arm)
        x,y=(row["arms"][k] for k in arms)
        row["pair_token_differences"]=sum(a!=b for a,b in zip(x["produced"],y["produced"]))
        summary["comparisons"].append(row)
        print("live",idx,{k:(len(v["steps"]),v["wrong"],round(v["seconds"],3)) for k,v in row["arms"].items()},flush=True)
        (here/"live_tree.json").write_text(json.dumps(summary,indent=2))
