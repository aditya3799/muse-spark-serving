"""CPU source allocation accounting and runtime-state recovery contracts."""
from __future__ import annotations
import ast
from dataclasses import dataclass, fields
import dataclasses
import hashlib
import itertools
import json
from pathlib import Path
import threading
from types import SimpleNamespace as NS
from typing import Optional, Dict
import torch
from target_verify_runtime_state import TargetVerifyStates
import prefix_width_scaffold as source

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SRT=ROOT/'sglang/python/sglang/srt'


def extract(path,name,cls=None):
    tree=ast.parse(path.read_text())
    body=tree.body if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body
    return next(n for n in body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)


def allocation_accounting():
    path=SRT/'layers/attention/flashattention_backend.py'
    ns={'torch':torch,'maybe_create_verify_mask':lambda **kw:None}
    source.load([extract(path,'init_cuda_graph_state','FlashAttentionBackend')],ns)
    rows=[]
    for width in (2,4,6,8):
        backend=NS(max_context_len=131072,page_size=1,device='meta',
            _get_scheduler_metadata=object(),use_mla=False,local_attn_builder=None,
            use_sliding_window_kv_pool=True,topk=1,speculative_num_draft_tokens=width,
            is_draft_runner=False,skip_prefill=False,is_encoder_decoder=False)
        ns['init_cuda_graph_state'](backend,32,32*width)
        tensors={}
        def visit(obj,path):
            if isinstance(obj,torch.Tensor): tensors[id(obj)]=(path,obj.numel()*obj.element_size())
            elif isinstance(obj,dict):
                for key,value in obj.items(): visit(value,path+'.'+str(key))
        visit(vars(backend),'backend')
        rows.append({'width':width,'metadata_bytes':sum(x[1] for x in tensors.values()),
            'page_table_bytes':sum(n for key,n in tensors.values() if 'page_table' in key),
            'input_embeds_bytes':32*width*8192*2,'aux_output_bytes':32*width*40960*2,
            'tensor_allocations':len(tensors)})
    return {'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'states':rows,
        'additional_metadata_bytes_for_2_4_6':sum(r['metadata_bytes'] for r in rows[:-1]),
        'additional_input_and_aux_bytes_for_2_4_6':sum(r['input_embeds_bytes']+r['aux_output_bytes'] for r in rows[:-1]),
        'limits':['Meta-device logical tensor bytes only, no allocator fragmentation/cubin/graph workspace',
                  'Same target weights and KV allocations; no new model',
                  'GraphSharedOutput shares existing max-row logits; not charged again',
                  'Other small input buffers omitted from input/aux subtotal',
                  'Runtime +2GB peak rule still requires actual startup measurement']}


def main():
    default=NS(speculative_num_draft_tokens=8)
    runner=NS(attn_backend=default,decode_cuda_graph_runner=object(),init_new_workspace=False,is_draft_worker=False)
    for key in ('model','req_to_token_pool','token_to_kv_pool','token_to_kv_pool_allocator','kv_index_translator','prefill_cuda_graph_runner','graph_shared_output'):
        setattr(runner,key,object())
    global_spec=NS(speculative_num_draft_tokens=8)
    draft=NS(block_size=8,attn_backend=object(),graph=object())
    snapshot=(vars(draft).copy(),vars(global_spec).copy())
    manager=TargetVerifyStates(runner,8,'forward')
    events=[]
    def backend_factory(mr):
        mr.init_new_workspace=True
        return type('FlashAttentionBackend',(),{} )()
    def good_backend(mr):
        b=backend_factory(mr)
        b.fa_impl_ver=3;b.target_verify_metadata={};b._verify_mask=None
        b.speculative_num_draft_tokens=8
        return b
    def graph_factory(mr,backend,width):
        assert mr.attn_backend is backend and backend.speculative_num_draft_tokens==width
        assert global_spec.speculative_num_draft_tokens==8 and draft.block_size==8
        return NS(attn_backend=backend,captured_req_width=width,model_runner=mr)
    for width in (2,4,6): manager.build(width,good_backend,graph_factory)
    checks=0
    for width in (2,8,4,6,2,4,8):
        with manager.use(width,'forward') as state:
            assert runner.attn_backend is state.backend
            assert runner.decode_cuda_graph_runner is state.graph
            checks+=1
        assert runner.attn_backend is default and not manager.active
        assert snapshot==(vars(draft),vars(global_spec))
    negatives=[]
    def reject(name,fn,exception=RuntimeError):
        try: fn()
        except exception: negatives.append(name)
        else: raise AssertionError(name)
    def cross_stream():
        with manager.use(2,'other'): pass
    reject('cross_stream',cross_stream)
    def nested():
        with manager.use(2,'forward'):
            with manager.use(4,'forward'): pass
    reject('nested',nested)
    def consumer_error():
        with manager.use(2,'forward'): raise RuntimeError('consumer failed')
    reject('consumer_exception_restore',consumer_error)
    assert runner.attn_backend is default and not manager.active
    def bad_graph(mr,b,w): raise RuntimeError('capture failed')
    reject('capture_exception_restore',lambda:manager.build(3,good_backend,bad_graph))
    assert 3 not in manager.states and not runner.init_new_workspace and runner.attn_backend is default
    badowner=runner.token_to_kv_pool;runner.token_to_kv_pool=object()
    reject('replaced_kv_owner',lambda:manager.seal_startup())
    runner.token_to_kv_pool=badowner
    def other_thread():
        try: manager.seal_startup()
        except RuntimeError: negatives.append('other_host_thread')
    t=threading.Thread(target=other_thread);t.start();t.join()
    manager.seal_startup()
    reject('late_capture_after_kv_sizing',lambda:manager.build(3,good_backend,graph_factory))
    report={'status':'CPU_TARGET_ONLY_STATE_CONTRACT','switches':checks,'negative_controls':negatives,
            'global_draft_width_unchanged':True,'allocation_accounting':allocation_accounting(),
            'holds':['Real graph runner construction and model execution','Weight-update/recapture integration across all states','No adaptive policy; rank-consistent choice required'],
            'source':hashlib.sha256((HERE/'target_verify_runtime_state.py').read_bytes()).hexdigest()}
    (HERE/'TARGET_ONLY_STATE_CPU.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
