"""Exercise the frozen logits dispatch statements and patched counterparts for greedy and fallback scoring."""
import ast
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from package import modify, ROOT, LP

here = Path(__file__).resolve().parent
source = (ROOT/"sglang"/LP).read_text(encoding="utf-8")
worker = (ROOT/"sglang/python/sglang/srt/speculative/dflash_worker_v2.py").read_text(encoding="utf-8")

def method(source,name):
    return next(node for node in ast.walk(ast.parse(source)) if isinstance(node,ast.FunctionDef) and node.name==name)

def compile_dispatch(src):
    block=method(src,"_get_logits").body
    start=next(i for i,n in enumerate(block) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=="can_fuse_gather" or isinstance(n,ast.If) and ast.unparse(n.test)=="self.logit_scale is not None")
    end=next(i for i,n in enumerate(block) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=="used_tp_lm_head_all_to_all")
    f=ast.parse("def dispatch(self,logits,logits_metadata,use_logits_buffer=True): pass").body[0]
    f.body=ast.parse("local_hidden_states=lm_head=None").body+block[start:end]+ast.parse("return logits").body
    functions=[f,method(src,"is_deferred_shard"),method(src,"finish_deferred_gather"),method(worker,"_can_skip_verify_logits")]
    scope={}
    exec(compile(ast.fix_missing_locations(ast.Module(body=ast.parse("from __future__ import annotations").body+functions,type_ignores=[])),"<parent dispatch>","exec"),scope)
    return scope

def run(f, raw, *, cap, shard, deferred, verify, scale, mode, selector, rounds):
    proc=SimpleNamespace(logit_scale=scale,defer_verify_gather=deferred,vocab_size=16,_selector_sample=selector)
    proc._can_gather_softcap_fused=lambda *args:cap
    proc._gather_softcap_fused=lambda logits,*args,**kwargs:torch.tanh(logits/20.)*20.
    meta=SimpleNamespace(forward_mode=SimpleNamespace(is_target_verify=lambda:verify))
    info=SimpleNamespace(is_all_greedy=mode!="sample",has_custom_logit_processor=mode=="custom",acc_linear_penalties=1 if mode=="penalty" else None,
                         penalizer_orchestrator=SimpleNamespace(is_required=mode=="required"),grammar_mask=1 if mode=="mask" else None,
                         logit_bias=1 if mode=="bias" else None,sampling_mask_batch_indices=1 if mode=="indices" else None)
    batch=SimpleNamespace(has_grammar=mode=="grammar",return_logprob=mode=="logprob")
    if mode=="absent":info=None
    x=torch.empty_like(raw)
    values=[]
    for i in range(rounds):
        x.copy_(raw*(1 if i==0 else -3))
        v=f["dispatch"](proc,x,meta)
        # DFlash finishes a deferred result only after a target verify forward.
        if verify and cap and f["is_deferred_shard"](proc,v):
            if f["_can_skip_verify_logits"](proc,batch,info):
                if f is candidate and scale is not None:v=v*scale
                v=(torch.tanh(v/20)*20).argmax(-1)
            else:
                v=f["finish_deferred_gather"](proc,v)
        values.append(v.clone())
    return values

if __name__=="__main__":
    parent=compile_dispatch(source)
    candidate=compile_dispatch(modify(source))
    g=torch.Generator().manual_seed(8471)
    cases=0
    for cap,shard,deferred,verify,scale in itertools.product((False,True),(False,True),(False,True),(False,True),(None,.1767766952966369)):
        raw=torch.randn((5,8 if shard else 16),generator=g)
        for mode in ("greedy","absent","sample","custom","penalty","required","mask","bias","indices","grammar","logprob"):
            for selector in (None,True):
                args=dict(cap=cap,shard=shard,deferred=deferred,verify=verify,scale=scale,mode=mode,selector=selector,rounds=2)
                a,b=(run(f,raw,**args) for f in (parent,candidate))
                assert all(torch.equal(x,y) for x,y in zip(a,b)),args
                cases+=1
    result=dict(cases=cases,rounds_per_case=2,all_equal=True,scope="Exact pinned and patched decision statements; local CPU math stands in for CUDA kernels and TP gathering.")
    (here/"dispatch.json").write_text(json.dumps(result,indent=2))
    print(result)