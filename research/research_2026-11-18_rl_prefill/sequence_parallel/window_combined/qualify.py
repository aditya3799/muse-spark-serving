"""Replay the combined captured-body and eager-wrapper ownership rules on real tensors."""
from __future__ import annotations
import ast,copy,json,sys,types
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch
import torch
H=Path(__file__).resolve().parent;ROOT=H.parents[2]
def functions(path,cls,method,bindings):
    nodes=ast.parse(path.read_text(encoding='utf-8-sig')).body
    if cls:nodes=next(x.body for x in nodes if isinstance(x,ast.ClassDef) and x.name==cls)
    fn=next(x for x in nodes if getattr(x,'name',False)==method)
    fn.decorator_list=[]
    d=dict(torch=torch,**bindings)
    exec(compile(ast.Module(ast.parse('from __future__ import annotations').body+[fn],[]),str(path),'exec'),d)
    return d[method]
sp=functions(H/'layernorm_sp.py',None,'runs_sp',dict(layernorm_sp_enabled=lambda:True,get_flags=lambda:S(sp=S(replicated_muse_experts=True)),get_parallel=lambda:S(tp_group=S(world_size=8))))
module=types.ModuleType('sglang.srt.model_executor.forward_batch_info');module.ForwardMode=S(EXTEND='extend')
tree=ast.parse((H/'prefill_cuda_graph_runner.py').read_text())
method=next(f for f in ast.walk(tree) if isinstance(f,ast.FunctionDef) and f.name=='_execute_body_capture')
choice=next(n for n in method.body if isinstance(n,ast.If) and ast.unparse(n.test)=='full_path' and 'tail_batch' in ast.unparse(n))
setter=compile(ast.Module([choice],[]),'<pinned runner tail_batch>','exec')
def tail(mode,request,static):
    if mode=='eager':return request
    ns=dict(full_path=mode=='full',forward_batch=request,static_forward_batch=static,copy=copy)
    exec(setter,ns);return ns['tail_batch']
cases=[]
with patch.dict(sys.modules,{'sglang.srt.model_executor.forward_batch_info':module}):
    for kind,raw,padded in [('eager',753,753),('eager',512,512),('full',753,768),('bcg',753,768)]:
        for embeddings in [False,True]:
            static=S(attn_tp_sequence_sharded=padded%8==0,forward_mode='extend')
            for i,omit in enumerate([False,True,False,True]):
                # Mark itself is attached only to the live request by the DFlash TP worker.
                request=S(forward_mode='extend',attn_tp_sequence_sharded=padded%8==0)
                if omit:request.dflash_skip_prefill_aux=True
                fb=tail(kind,request,static);use_sp=padded%8==0 and padded>=512
                size=padded//8 if use_sp else padded
                # BCG's eager outer tail sees the padded token IDs.
                usersize=padded if kind=='bcg' else raw
                data=torch.arange(padded*4,dtype=torch.float32).view(padded,4)+i
                taps=[(data+100*j).bfloat16() for j in range(5)]
                trace=[];packed=torch.cat(taps,1)
                def gather(x,num_tokens):
                    source=packed if x.shape[1]==20 else data
                    assert torch.equal(x,source[:size])
                    trace.append(x.shape[1]);return source[:num_tokens]
                def model(ids,pos,b,**kwargs):
                    if kind!='eager' or not omit:return data[:size],[x[:size] for x in taps]
                    return data[:size]
                def logits(ids,hidden,head,b,aux):
                    if ids is not None and sp(b.forward_mode,usersize,returned_rows=hidden.shape[0]):hidden=gather(hidden,usersize)
                    assert torch.equal(hidden[:raw],data[:raw])
                    if omit:assert aux is None
                    else:
                        if isinstance(aux,list):aux=torch.cat(aux,1)
                        assert torch.equal(aux[:raw],packed[:raw])
                    return hidden[torch.tensor([0,raw//2,raw-1])]
                wrap=functions(H/'muse_spark.py','MuseSparkTextForCausalLM','forward',dict(layernorm_sp=S(runs_sp=sp,sp_exit_gather=gather),get_forward=lambda:S(set=lambda *a:None),pack_aux_hidden_states=lambda z:torch.cat(z,1)))
                result=wrap(S(model=model,capture_aux_hidden_states=True,logits_processor=logits,lm_head=None),None if embeddings else torch.arange(usersize),None,fb,input_embeds=torch.zeros(usersize,4) if embeddings else None)
                assert torch.equal(result,data[[0,raw//2,raw-1]])
                assert trace.count(20)==int(use_sp and not omit) and trace.count(4)==int(use_sp)
                assert not hasattr(static,'dflash_skip_prefill_aux')
                cases.append(dict(mode=kind,embeddings=embeddings,omit=omit,raw=raw,padded=padded,gathered_widths=trace))
out={'cases':cases,'coverage':'Actual combined wrapper, runner tail ownership and SP dispatch. Mock collective gathers independently populated rank rows. FP32 hidden/logit features and BF16 taps. Frozen workers unchanged. No H200 timing.'}
(H/'ownership.json').write_text(json.dumps(out,indent=2));print('passed',len(cases))

