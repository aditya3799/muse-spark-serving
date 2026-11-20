"""Exercise actual overlay worker tail with CPU tensors and explicit fake models.
The entire post-proposal DFlash worker control flow is source-extracted. The
real target/draft models and CUDA kernels are not run; no numerical-model claim.
"""
from __future__ import annotations
import ast
import hashlib
import itertools
from functools import lru_cache
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS, ModuleType
from typing import Optional
import torch
torch.set_num_threads(1)
import prefix_width_scaffold as scaffold
import dflash_prefix_verification as support
from target_verify_runtime_state import TargetVerifyStates
from target_prefix_graph_runtime import target_prefix_scope

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SRT = ROOT/'sglang/python/sglang/srt'
OVERLAY = HERE/'graph_prefix_overlay/python/sglang/srt'


def find(path, name, cls=None):
    tree = ast.parse(path.read_text())
    body = tree.body if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body
    return next(n for n in body if isinstance(n, (ast.FunctionDef,ast.ClassDef)) and n.name==name)


def accept_cpu_kernel(**kw):
    candidates, predicted = kw['candidates'], kw['target_top1']
    match = candidates[:,1:] == predicted[:,:-1]
    a = match.to(torch.int32).cumprod(1).sum(1).to(torch.int32)
    b = predicted[torch.arange(len(a)), a.long()]
    out = kw['out_tokens_out']
    assert out.shape == candidates.shape and out.is_contiguous()
    out[:,:-1].copy_(candidates[:,1:]); out[:,-1].zero_()
    out.scatter_(1, a.long()[:,None], b[:,None])
    kw['accept_lens_out'].copy_(a)
    kw['commit_lens_out'].copy_(a+1)
    kw['bonus_ids_out'].copy_(b)
    kw['new_seq_lens_out'].copy_(kw['prefix_lens']+a+1)


@lru_cache(maxsize=2)
def get_worker_class(candidate):
    path=(OVERLAY if candidate else SRT)/'speculative/dflash_worker_v2.py'
    ns = dict(scaffold.namespace)
    ns.update({'torch':torch, 'Optional':Optional, 'SIMULATE_ACC_LEN':0,
      'resolve_prefix_verification_width':support.resolve_prefix_verification_width,
      'target_prefix_scope':lambda worker,width:target_prefix_scope(worker,width,stream_key=17),
      'next_prefix_accept_buffers':support.next_prefix_accept_buffers,
      '_is_all_greedy':lambda info: info is None or info.is_all_greedy,
      'SpecTpSyncSite':NS(DFLASH_ACCEPT_GREEDY='greedy'),
      '_compute_dflash_accept_bonus_triton_unchecked':accept_cpu_kernel,
      'SpeculativeSamplingMaskCapture':NS(from_logits=lambda *a,**kw:None),
      'apply_dflash_verify_logits_adjustments':lambda **kw:None,
      'GenerationBatchResult':lambda **kw:NS(**kw),
      'get_parallel':lambda:parallel,
      '_is_npu':False})
    methods=['_ensure_accept_bonus_buffers','_next_accept_bonus_buffers','_accept_block','_can_skip_verify_logits']
    scaffold.load([find(path,'_commit_accept')]+[find(path,n,'DFlashWorkerV2') for n in methods],ns)
    kernel_path=ROOT/'sglang/python/sglang/kernels/ops/speculative/dflash.py'
    scaffold.load([find(kernel_path,'compute_dflash_correct_drafts_and_bonus')],ns)
    method=find(path,'forward_batch_generation','DFlashWorkerV2')
    if candidate:
        first=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='verify_block_size' for t in n.targets))
    else:
        # First custom_mask assignment is exactly the target-verify boundary.
        first=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='custom_mask' for t in n.targets))
    names=['self','batch','bs','block_size','draft_tokens','positions','positions_2d','verify_out_cache_loc','verify_out_cache_loc_2d','kv_loc_plan','prefix_lens','draft_input','on_publish','grammar_tree','grammar_barrier']
    tail=ast.FunctionDef(name='tail',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n) for n in names],kwonlyargs=[],kw_defaults=[],defaults=[]),body=method.body[first:],decorator_list=[])
    scaffold.load([tail],ns)
    attributes={name:ns[name] for name in methods}
    attributes['tail']=ns['tail']
    return type('ExtractedWorker',(),attributes)


parallel=NS(attn_dp_enabled=False,attn_cp_size=1,attn_dcp_size=1,pp_size=1)
original_fa3=scaffold.namespace['init_forward_metadata']
patched_ns=dict(scaffold.namespace)
scaffold.load([find(OVERLAY/'layers/attention/flashattention_backend.py','init_forward_metadata','FlashAttentionBackend')],patched_ns)

class FlashAttentionBackend(scaffold.Backend):
    fa_impl_ver=3
    init_forward_metadata=patched_ns['init_forward_metadata']


def make_fixture(bs, width, generation, candidate, triton_model=True, fallback=False, cpu_lengths=True, padded=False, trim=False, reject=False):
    table=torch.arange(40*256,dtype=torch.int64).view(40,256)+100
    reqs=torch.arange(bs,dtype=torch.int64).flip(0)
    prefix=torch.arange(bs,dtype=torch.int64)+40+generation
    positions=prefix[:,None]+torch.arange(8)
    ids=torch.arange(bs*8,dtype=torch.int64).view(bs,8)+100
    slots=table[reqs[:,None],positions]
    reader=scaffold.Reader(table,True)
    plan=scaffold.SourcePlan(reqs,slots.flatten(),reader)
    backend=FlashAttentionBackend(8,table,reader,swa=True)
    if not candidate:
        backend.init_forward_metadata=original_fa3.__get__(backend,type(backend))
    runner=NS(attn_backend=backend,decode_cuda_graph_runner=scaffold.Graph(8,backend),translator=reader,
              model_config=NS(is_encoder_decoder=False),lora_manager=None)
    expected_width=8 if fallback else width
    sampling=NS(is_all_greedy=True,penalizer_orchestrator=None,has_custom_logit_processor=False,
                acc_linear_penalties=None,grammar_mask=None,logit_bias=None,sampling_mask_batch_indices=None)
    batch=NS(forward_mode=scaffold.ForwardMode.DECODE,seq_lens=prefix,seq_lens_cpu=prefix.clone() if cpu_lengths else None,
             seq_lens_sum=int(prefix.sum()),req_pool_indices=reqs,out_cache_loc=slots.flatten(),encoder_lens=None,
             sampling_info=sampling,has_grammar=False,return_logprob=False,is_extend_in_batch=False)
    events=[]
    calls=[]
    def target_forward(**kw):
        fb=kw['forward_batch']; w=fb.spec_info.draft_token_num
        assert w==expected_width and fb.input_ids.numel()==bs*w
        assert fb.positions.tolist()==positions[:,:w].flatten().tolist()
        assert torch.equal(fb.out_cache_loc,reader.transform(slots[:,:w].flatten(),scaffold.IdSpaceKind.FULL))
        assert torch.equal(runner.attn_backend.forward_metadata.cache_seqlens_int32,(prefix+w).to(torch.int32))
        if candidate: assert worker._prefix_target_states.active
        selected=fb.input_ids.view(bs,w)
        predictions=torch.full((bs,w),900,dtype=torch.int64)
        if w>1: predictions[:,:-1]=selected[:,1:]
        accepts=[]
        for row in range(bs):
            a=(row+generation)%w
            accepts.append(a)
            if a<w-1: predictions[row,a]=950+row
        logits=torch.full((bs*w,1024),-1.0)
        logits[torch.arange(bs*w),predictions.flatten()]=10.0
        hidden=torch.arange(bs*w*13,dtype=torch.float32).reshape(bs*w,13)+generation*10000
        calls.append({'ids_ptr':fb.input_ids.data_ptr(),'ids':selected.clone(),'predictions':predictions,
                      'hidden':hidden.clone(),'accept':torch.tensor(accepts),'width':w})
        events.append('target')
        output=NS(next_token_logits=logits,hidden_states=hidden)
        if padded:
            output.next_token_logits=torch.cat([logits,torch.full((5,1024),float('nan'))])
            output.hidden_states=torch.cat([hidden,torch.full((5,13),float('nan'))])
        if trim:
            source_trim(fb,output,bs)
        return NS(logits_output=output,
                  can_run_cuda_graph=True,routed_experts_output=None,indexer_topk_output=None)
    target_worker=NS(model_runner=runner,forward_batch_generation=target_forward)
    Worker=get_worker_class(candidate)
    worker=Worker()
    for name,value in {'block_size':8,'device':torch.device('cpu'),'target_worker':target_worker,
       'draft_model_runner':NS(kv_index_translator=reader),'_need_mamba_verify_commit':False,'lilicorr':None,
       '_is_domino':False,'_selector_sample':None,'_target_logits_processor':None,'_use_triton_accept_bonus':triton_model,
       '_accept_bonus_buffer_cap':0,'_accept_bonus_buffer_slot':0,'_accept_len_buf':None,
       '_tp_sync':NS(sync=lambda *args:None)}.items(): setattr(worker,name,value)
    if candidate:
        runner.init_new_workspace=False;runner.is_draft_worker=False
        for key in ('model','req_to_token_pool','token_to_kv_pool','token_to_kv_pool_allocator','kv_index_translator','prefill_cuda_graph_runner','graph_shared_output'):
            setattr(runner,key,object())
        manager=TargetVerifyStates(runner,8,None)
        def make_backend(mr):
            b=FlashAttentionBackend(8,table,reader,swa=True)
            b.target_verify_metadata={};b._verify_mask=None
            return b
        def make_graph(mr,b,w):
            g=scaffold.Graph(w,b);g.attn_backend=b;g.model_runner=mr
            return g
        if width!=8: manager.build(width,make_backend,make_graph)
        manager.seal_startup();worker._prefix_target_states=manager
    worker._draft_write_ids=lambda p:p.write_ids(reader)
    worker._make_next_draft_input_decode=lambda **kw:NS(**kw)
    def append(**kw):
        if candidate: assert worker._prefix_target_states.active
        events.append('append')
        w=expected_width
        assert kw['target_hidden'].shape==(bs*w,13)
        assert torch.equal(kw['target_hidden'],calls[-1]['hidden'])
        assert torch.equal(kw['cache_loc'],reader.transform(slots[:,:w].flatten(),scaffold.IdSpaceKind.FULL))
        assert kw['cache_loc_2d'].shape==(bs,w)
        assert torch.equal(kw['positions'],positions[:,:w].flatten())
        assert torch.equal(kw['commit_lens'],calls[-1]['accept']+1)
    worker._append_target_hidden_to_draft_kv_by_loc=append
    if candidate: support.configure_prefix_verification(worker,width)
    if fallback: runner.lora_manager=object()
    def publish(lengths):
        assert torch.equal(lengths,prefix+calls[-1]['accept']+1)
        if candidate: assert worker._prefix_target_states.active
        events.append('publish')
    draft_input=NS(nxt_kv_lens_cpu=prefix.cpu()+16,nxt_kv_lens_sum=int((prefix+16).sum()))
    try:
        result=worker.tail(batch,bs,8,ids,positions.flatten(),positions,slots.flatten(),slots,plan,prefix,draft_input,publish,None,None)
    except ValueError as error:
        if candidate:
            assert not manager.active and runner.attn_backend is backend
        if reject:
            assert 'retain padding' in str(error)
            return {'rejected':True}
        raise
    if candidate:
        assert not manager.active and runner.attn_backend is backend
    assert not reject, 'padded outputs were admitted'
    assert (batch.seq_lens_cpu is not None)==cpu_lengths
    assert events==['target','publish','append']
    assert result.logits_output.hidden_states is None
    assert result.speculative_num_draft_tokens==8
    stride=getattr(result,'speculative_output_stride',None)
    assert stride==(None if expected_width==8 else expected_width)
    assert torch.equal(result.accept_lens,calls[-1]['accept']+1)
    tokens=result.next_token_ids.view(bs,expected_width)
    for row,a in enumerate(calls[-1]['accept'].tolist()):
        expected=ids[row,1:a+1].tolist()+[int(calls[-1]['predictions'][row,a])]
        assert tokens[row,:a+1].tolist()==expected
    if expected_width==8:
        assert calls[-1]['ids_ptr']==ids.data_ptr()
        assert not hasattr(worker,'_prefix_accept_buffers')
    else:
        assert getattr(worker,'_prefix_accept_buffers',None) is not None if triton_model else True
    return {'tokens':tokens.tolist(),'accept':result.accept_lens.tolist(),
            'bonus':result.next_draft_input.bonus_tokens.tolist(),'seq':result.new_seq_lens.tolist()}


def build_source_trim():
    path=SRT/'model_executor/forward_batch_info.py'
    fn=find(path,'post_forward_mlp_sync_batch','ForwardBatch')
    branch=next(n for n in ast.walk(fn) if isinstance(n,ast.If) and ast.unparse(n.test)=='self.forward_mode.is_target_verify()')
    trim=ast.FunctionDef(name='trim',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n) for n in ('self','logits_output','bs')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=branch.body,decorator_list=[])
    ns={}
    scaffold.load([trim],ns)
    return ns['trim']

source_trim=build_source_trim()

def main():
    module=ModuleType('sglang.srt.speculative.spec_utils')
    module.prepare_mamba_track_for_verify=lambda batch:None
    sys.modules[module.__name__]=module
    report={'status':'CPU_GRAPH_SCOPED_SOURCE_WORKER_TAIL_PASS','cases':0,'default8_reference_comparisons':0,
            'fallback_cases':0,'fakes':['Target logits/aux tensors','CPU model of Triton accept kernel; separate pinned CPU accept fallback tested','ForwardBatch init packaging','Fake Graph.load_batch invokes extracted actual FA3 metadata with CPU builder; not actual DecodeCudaGraphRunner/model','No-op logits adjustment and sampling-mask paths restricted to plain greedy','No real model or CUDA execution'],
            'holds':['Model/attention numeric equivalence across shapes','Actual rank-consistent runtime configuration','Graph specialization and memory','Full serving correctness/performance']}
    for bs,w,g,kernel in itertools.product((1,2,3,8,17,32),(1,2,4,6,8),range(3),(False,True)):
        result=make_fixture(bs,w,g,True,kernel)
        report['cases']+=1
        if w==8:
            baseline=make_fixture(bs,8,g,False,kernel)
            assert result==baseline
            report['default8_reference_comparisons']+=1
    for bs,w in itertools.product((1,3,32),(1,2,4,6)):
        actual=make_fixture(bs,w,1,True,True,True)
        baseline=make_fixture(bs,8,1,False,True)
        assert actual==baseline
        report['fallback_cases']+=1
    report['padded_post_forward_cases']=0
    report['untrimmed_padding_rejections']=0
    report['no_cpu_length_cases']=0
    for bs,w in itertools.product((1,3,17),(1,2,4,6)):
        make_fixture(bs,w,2,True,True,cpu_lengths=False)
        report['no_cpu_length_cases']+=1
        make_fixture(bs,w,2,True,True,padded=True,trim=True)
        report['padded_post_forward_cases']+=1
        make_fixture(bs,w,2,True,True,padded=True,reject=True)
        report['untrimmed_padding_rejections']+=1
    report['sources']={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'dflash_prefix_verification.py',HERE/'graph_prefix_overlay.diff',OVERLAY/'speculative/dflash_worker_v2.py',OVERLAY/'layers/attention/flashattention_backend.py']}
    (HERE/'GRAPH_SCOPED_WORKER_TAIL_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
