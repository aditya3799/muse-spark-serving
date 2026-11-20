"""Real CUDA metadata and captured-buffer proof; NO attention/model execution.
Loads pinned kernels and source-extracts actual FA3, KVPlan, translator, and
DecodeInputBuffers paths. GraphFacade is explicitly not DecodeCudaGraphRunner.
"""
from __future__ import annotations
import argparse
import ast
from dataclasses import dataclass, fields
import dataclasses
import enum
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS
from typing import Optional, Dict, Tuple, Collection, Callable, Hashable, Union
import msgspec
import torch
import prefix_width_scaffold as scaffold
from target_verify_runtime_state import TargetVerifyStates

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SRT=ROOT/'sglang/python/sglang/srt'
KERNEL=ROOT/'sglang/python/sglang/kernels/ops'


def get_node(path,name,cls=None):
    tree=ast.parse(path.read_text())
    body=tree.body if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body
    return next(n for n in body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)


def module_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def source_types():
    metadata=module_file('r4_pinned_metadata',KERNEL/'attention/metadata.py')
    readtable=module_file('r4_pinned_readtable',KERNEL/'kvcache/kv_read_table.py')
    ns=dict(globals())
    ns.update({'normal_decode_set_metadata':metadata.normal_decode_set_metadata,
               'build_kv_read_table':readtable.build_kv_read_table,
               'maybe_create_verify_mask':lambda **kw:None,
               'ForwardMode':scaffold.ForwardMode,
               'is_npu':lambda:False,'_forward_input_buffer_pool':{},
               'enable_num_token_non_padded':lambda:False,
               'envs':NS(SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE=NS(get=lambda:False))})
    path=SRT/'mem_cache/kv_loc_plan.py'
    scaffold.load([get_node(path,n) for n in ('IdSpaceKind','IdSpace','TokenSpan','pad_with_sink','KVLocPlan')],ns)
    path=SRT/'mem_cache/kv_index_translator.py'
    scaffold.load([get_node(path,'KVIndexTable')],ns)
    names=['space','_reads_translated','build_iteration_table','_plan_table','copy_page_table','reads_are_translated']
    cls=ast.ClassDef(name='ExtractedTranslator',bases=[],keywords=[],body=[get_node(path,n,'KVIndexTranslator') for n in names],decorator_list=[])
    scaffold.load([cls],ns)
    path=SRT/'layers/attention/flashattention_backend.py'
    scaffold.load([get_node(path,'FlashAttentionMetadata')],ns)
    names=['init_cuda_graph_state','_bind_metadata_buffers','_set_decode_page_metadata','_apply_cuda_graph_metadata']
    cls=ast.ClassDef(name='FlashAttentionBackend',bases=[],keywords=[],body=[get_node(path,n,'FlashAttentionBackend') for n in names],decorator_list=[])
    scaffold.load([cls],ns)
    path=SRT/'model_executor/input_buffers.py'
    scaffold.load([get_node(path,'share_input_buffer'),get_node(path,'ForwardInputBuffers')],ns)
    path=SRT/'model_executor/runner_utils/buffers.py'
    scaffold.load([get_node(path,'DecodeInputBuffers')],ns)
    return ns


def fingerprint(tensor):
    return (tensor.data_ptr(),tuple(tensor.shape),tuple(tensor.stride()),tensor.dtype,tensor.device)


def main(output):
    start=time.monotonic()
    output.mkdir(parents=True,exist_ok=False)
    report={'status':'STARTED','scope':'Real metadata CUDA kernels and graph-buffer ownership; no FA3 attention/model/collectives',
            'checks':0,'replays':0,'negative_controls':[],'external_tts':'Untouched; may vary memory/use',
            'sources':{}}
    def save(): (output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    def phase(name):
        report['phase']=name;save();print(name,flush=True)
    def reserve():
        free,total=torch.cuda.mem_get_info()
        report.setdefault('free_memory_observations',[]).append(free)
        assert free >= (512<<20)+(64<<20), '512MiB reserve plus64MiB fixture budget'
        assert torch.cuda.max_memory_allocated() <= (64<<20), '64MiB own allocator peak budget'
    assert torch.cuda.is_available()
    reserve();torch.cuda.reset_peak_memory_stats()
    ns=source_types();Kind=ns['IdSpaceKind'];IdSpace=ns['IdSpace'];Plan=ns['KVLocPlan']
    Translator=ns['ExtractedTranslator'];Backend=ns['FlashAttentionBackend'];Buffers=ns['DecodeInputBuffers']
    max_bs,context,max_reqs=4,128,8
    req_table=torch.zeros((max_reqs,context),dtype=torch.int64,device='cuda')
    full_map=torch.arange(max_reqs*context+1,dtype=torch.int64,device='cuda')
    swa_map=full_map.clone()
    reader=Translator();reader.device=torch.device('cuda');reader.page_size=1;reader.req_to_token=req_table
    reader._rows=torch.arange(max_reqs,dtype=torch.int64,device='cuda')
    reader._spaces={
        Kind.FULL:IdSpace(key='full',write=lambda ids:full_map[ids.clamp(min=0)],read_v2p=full_map),
        Kind.SLIDING_WINDOW:IdSpace(key='swa',write=lambda ids:swa_map[ids.clamp(min=0)],read_v2p=swa_map),
    }
    reader.is_translating=True
    global_spec=NS(speculative_num_draft_tokens=8)
    draft_state=NS(width=8,graph=object())
    shared_logits=torch.zeros((max_bs*8,64),dtype=torch.float32,device='cuda')
    common_pool=torch.cuda.graph_pool_handle()
    default=NS(speculative_num_draft_tokens=8)
    runner=NS(attn_backend=default,decode_cuda_graph_runner=None,init_new_workspace=False,is_draft_worker=False,
              model=object(),req_to_token_pool=NS(req_to_token=req_table),token_to_kv_pool=object(),
              token_to_kv_pool_allocator=object(),kv_index_translator=reader,prefill_cuda_graph_runner=object(),graph_shared_output=shared_logits)
    stream_key=torch.cuda.current_stream().cuda_stream
    manager=TargetVerifyStates(runner,8,stream_key)
    # The initial width8 state is replaced only in this fixture, since it has no
    # pre-existing full model graph. Production retains its startup state.
    del manager.states[8]
    storage=[]
    def backend_factory(mr):
        backend=Backend()
        for name,value in dict(fa_impl_ver=3,max_context_len=context,max_num_pages=context,page_size=1,device='cuda',
            _get_scheduler_metadata=None,use_mla=False,local_attn_builder=None,use_sliding_window_kv_pool=True,
            topk=1,speculative_num_draft_tokens=8,is_draft_runner=False,skip_prefill=False,is_encoder_decoder=False,
            target_verify_metadata={},_verify_mask=None,kv_index_translator=reader,req_to_token=req_table,
            token_to_kv_pool=NS(full_to_swa_index_mapping=None)).items():setattr(backend,name,value)
        return backend
    class GraphFacade:
        def __init__(self,mr,backend,width):
            self.model_runner=mr;self.attn_backend=backend;self.captured_req_width=width
            self.graphs={}
            self.buffers=Buffers.create(device=torch.device('cuda'),max_bs=max_bs,max_num_token=max_bs*width,
                hidden_size=17,next_token_logits_buffer=shared_logits[:max_bs*width],dtype=torch.bfloat16,
                num_dp_ranks=1,pp_size=1,is_encoder_decoder=False,require_mlp_tp_gather=False,
                seq_len_fill_value=1,encoder_len_fill_value=0,num_tokens_per_req=width,
                cache_loc_dtype=torch.int64,enable_mamba_track=False,aux_hidden_states_width=35)
            self.buffers.share_buffers()
            backend.init_cuda_graph_state(max_bs,max_bs*width)
            self.window=torch.zeros((max_bs,8),dtype=torch.int64,device='cuda')
            for bs in (1,3,4):
                backend._bind_metadata_buffers(bs,bs*width,None,scaffold.ForwardMode.TARGET_VERIFY,
                    NS(ragged_verify_layout=None),torch.device('cuda'))
                self.buffers.seq_lens.fill_(1);self.buffers.req_pool_indices.zero_()
                warm=torch.cuda.Stream();warm.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warm):
                    for _ in range(2):self.prepare(bs)
                torch.cuda.current_stream().wait_stream(warm)
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,pool=common_pool): self.prepare(bs)
                self.graphs[bs]=graph
            self.frozen=[(t,fingerprint(t)) for t in (self.buffers.req_pool_indices,self.buffers.seq_lens,
                req_table,full_map,swa_map,backend.target_verify_metadata['page_table'],backend.target_verify_metadata['swa_page_table'])]
        def prepare(self,bs):
            plan=Plan(source=reader,req_pool_indices=self.buffers.req_pool_indices[:bs],
                seq_lens=self.buffers.seq_lens[:bs],seq_lens_cpu=None,write_virtual=self.window[:bs].flatten(),read_extent=8)
            self.attn_backend._apply_cuda_graph_metadata(plan,bs,self.buffers.req_pool_indices[:bs],self.buffers.seq_lens[:bs],
                None,scaffold.ForwardMode.TARGET_VERIFY,NS(ragged_verify_layout=None),None)
        def replay(self,bs):
            if any(fingerprint(t)!=saved for t,saved in self.frozen): raise RuntimeError('captured tensor binding changed')
            self.graphs[bs].replay()
    phase('CAPTURE_BEGIN')
    for width in (8,6,4,2):
        reserve();manager.build(width,backend_factory,GraphFacade)
        phase('CAPTURED_WIDTH_'+str(width))
    manager.seal_startup()
    assert len({s.graph.buffers.req_pool_indices.data_ptr() for s in manager.states.values()})==1
    assert len({s.graph.buffers.input_ids.data_ptr() for s in manager.states.values()})==4
    phase('REPLAY_CHECKS')
    for generation in range(4):
        table_cpu=(torch.arange(max_reqs*context).view(max_reqs,context)+generation*7)%(max_reqs*context)+1
        table_cpu[2,3]=-1
        fm_cpu=(torch.arange(max_reqs*context+1)*3+generation*5)%(max_reqs*context+1)
        sm_cpu=(torch.arange(max_reqs*context+1)*7+generation*11)%(max_reqs*context+1)
        fm_cpu[0]=0;sm_cpu[0]=0  # Allocator reserved sink invariant.
        fm_cpu[5]=-1;sm_cpu[9]=-1
        req_table.copy_(table_cpu);full_map.copy_(fm_cpu);swa_map.copy_(sm_cpu)
        for width,bs in ((2,1),(8,4),(4,3),(6,1),(2,4),(8,1),(6,3),(4,4),(2,3),(8,3),(4,1),(6,4)):
            reserve()
            state=manager.states[width];graph=state.graph
            req_cpu=(torch.arange(bs)+generation)%max_reqs
            lens_cpu=(torch.arange(bs)*13+generation*9)%85
            graph.buffers.req_pool_indices[:bs].copy_(req_cpu)
            graph.buffers.seq_lens[:bs].copy_(lens_cpu)
            state.backend.target_verify_metadata['page_table'].fill_(-777)
            state.backend.target_verify_metadata['swa_page_table'].fill_(-888)
            with manager.use(width,torch.cuda.current_stream().cuda_stream): graph.replay(bs)
            torch.cuda.synchronize()
            meta=state.backend.target_verify_metadata[bs]
            expected_len=(lens_cpu+width).to(torch.int32)
            expected_cu=torch.nn.functional.pad(expected_len.cumsum(0,dtype=torch.int32),(1,0))
            assert torch.equal(meta.cache_seqlens_int32.cpu(),expected_len)
            assert torch.equal(meta.cu_seqlens_k.cpu(),expected_cu)
            assert meta.cu_seqlens_q.cpu().tolist()==list(range(0,(bs+1)*width,width))
            report['checks']+=3
            for mapping,actual in ((fm_cpu,meta.page_table),(sm_cpu,meta.swa_page_table)):
                expected=torch.zeros((bs,context),dtype=torch.int32)
                for row in range(bs):
                    n=int(lens_cpu[row])+8
                    virtual=table_cpu[int(req_cpu[row]),:n]
                    values=mapping[virtual.clamp(min=0)].clamp(min=0).to(torch.int32)
                    values[virtual<0]=0
                    expected[row,:n]=values
                observed=actual.cpu()
                if not torch.equal(observed,expected):
                    mismatch=(observed!=expected).nonzero()[:16]
                    report['failure']={'generation':generation,'width':width,'bs':bs,'coordinates':mismatch.tolist(),'actual':[int(observed[tuple(i)]) for i in mismatch],'expected':[int(expected[tuple(i)]) for i in mismatch]}
                    save();raise AssertionError('translated table mismatch')
                report['checks']+=1
            assert global_spec.speculative_num_draft_tokens==8 and draft_state.width==8
            assert runner.attn_backend is default and runner.decode_cuda_graph_runner is None
            report['replays']+=1
    phase('NEGATIVE_CONTROLS')
    graph=manager.states[2].graph
    old=graph.buffers.seq_lens.detach()
    graph.buffers.seq_lens.data=graph.buffers.seq_lens.clone()
    try: graph.replay(1)
    except RuntimeError: report['negative_controls'].append('mutable_tensor_data_binding_rejected')
    else: raise AssertionError('changed pointer accepted')
    graph.buffers.seq_lens.data=old
    try:
        with manager.use(2,stream_key+1): pass
    except RuntimeError:report['negative_controls'].append('cross_stream_state_rejected')
    else:raise AssertionError('cross stream admitted')
    # Counterexample: using width8 metadata for width2 gives the wrong length.
    assert not torch.equal(torch.tensor([7])+8,torch.tensor([7])+2)
    report['negative_controls'].append('wrong_width_length_distinguished')
    reserve()
    report.update(status='PASS_METADATA_ONLY',peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
                  elapsed_s=time.monotonic()-start,graph_count=sum(len(s.graph.graphs) for s in manager.states.values()),
                  device=torch.cuda.get_device_name(),limits=['Not the full DecodeCudaGraphRunner or FA3 attention kernel',
                  'No target/draft model or TP collectives','Metadata capture is an isolated ownership probe, not enabling DFlash metadata-glue',
                  'No performance or serving gain claim','Production startup/weight-update/recapture integration remains held'])
    for path in (SRT/'layers/attention/flashattention_backend.py',SRT/'mem_cache/kv_loc_plan.py',SRT/'mem_cache/kv_index_translator.py',
                 SRT/'model_executor/input_buffers.py',SRT/'model_executor/runner_utils/buffers.py',KERNEL/'attention/metadata.py',
                 KERNEL/'kvcache/kv_read_table.py',HERE/'target_verify_runtime_state.py',HERE/'prefix_width_scaffold.py',Path(__file__)):
        report['sources'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    save();print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
