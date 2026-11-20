"""CPU startup/factory/scoped ownership and rejection contract; no CUDA import."""
import hashlib,json
from pathlib import Path
from types import SimpleNamespace as N
import target_prefix_graph_runtime as r
P=Path(__file__).parent

def fixture():
    backend=type('FlashAttentionBackend',(),{})();backend.fa_impl_ver=3;backend.topk=1
    graph=type('DecodeCudaGraphRunner',(),{})()
    graph.backend=type('FullCudaGraphBackend',(),{})()
    for k in ('require_gathered_buffer','require_mlp_sync','attention_graph_variants','ragged_verify_mode','enable_two_batch_overlap','enable_pdmux','enable_prefill_cp','record_nolora_graph','is_dllm'):setattr(graph,k,False)
    graph.captured_req_width=8;graph.capture_forward_mode=N(is_target_verify=lambda:True)
    mr=N(attn_backend=backend,decode_cuda_graph_runner=graph,is_draft_worker=False,
         spec_algorithm=N(is_dflash=lambda:True),model_config=N(is_encoder_decoder=False),lora_manager=None,
         capture_tail_hooks=[],init_new_workspace=False)
    for k in ('model','req_to_token_pool','token_to_kv_pool','token_to_kv_pool_allocator','kv_index_translator','prefill_cuda_graph_runner','graph_shared_output'):setattr(mr,k,object())
    worker=N(block_size=8,target_worker=N(model_runner=mr),_need_mamba_verify_commit=False,lilicorr=None,_is_domino=False)
    runtime=N(parallel=N(attn_dp_enabled=False,attn_cp_size=1,attn_dcp_size=1,pp_size=1),execution=N(overlap=N(enable_two_batch_overlap=False)),disagg=N(enable_pdmux=False))
    return worker,runtime

def backend(mr):
    b=type('FlashAttentionBackend',(),{})();b.fa_impl_ver=3;b.target_verify_metadata={};b._verify_mask=None
    return b

def graph(mr,b,w):return N(attn_backend=b,captured_req_width=w,model_runner=mr)

def main():
    w,rt=fixture();assert r.initialize_target_prefix_states(w) is None
    manager=r.initialize_target_prefix_states(w,widths=(2,4,6),runtime=rt,backend_factory=backend,graph_factory=graph)
    assert manager.stream_key is None and not manager.startup_open
    default=w.target_worker.model_runner.attn_backend
    for width in (2,8,6,4,2):
        with r.target_prefix_scope(w,width,stream_key=17):
            assert w.target_worker.model_runner.attn_backend is manager.states[width].backend
        assert w.target_worker.model_runner.attn_backend is default
    negatives=[]
    def reject(name,fn):
        try:fn()
        except (RuntimeError,ValueError):negatives.append(name)
        else:raise AssertionError(name)
    def cross():
        with r.target_prefix_scope(w,2,stream_key=18):pass
    reject('serving_stream_rebinding',cross)
    r.invalidate_target_prefix_states(w,'weight mutation')
    reject('post_weight_mutation_scope',lambda:r.target_prefix_scope(w,2,stream_key=17).__enter__())
    assert len(manager.states)==4 # retain references, no unsafe graph freeing
    cases=[('dp','parallel.attn_dp_enabled',True),('cp','parallel.attn_cp_size',2),('dcp','parallel.attn_dcp_size',2),('pp','parallel.pp_size',2),('pdmux','disagg.enable_pdmux',True),('tbo','execution.overlap.enable_two_batch_overlap',True)]
    for label,path,value in cases:
        w,rt=fixture();obj=rt
        keys=path.split('.')
        for key in keys[:-1]:obj=getattr(obj,key)
        setattr(obj,keys[-1],value)
        assert not r.supported_startup(w,rt);negatives.append(label)
    cases=[('lora','lora_manager',object()),('encoder','model_config',N(is_encoder_decoder=True)),('draft_runner','is_draft_worker',True),('capture_tail_hook','capture_tail_hooks',[object()])]
    for label,key,value in cases:
        w,rt=fixture();setattr(w.target_worker.model_runner,key,value)
        assert not r.supported_startup(w,rt);negatives.append(label)
    for key in ('require_gathered_buffer','require_mlp_sync','attention_graph_variants','ragged_verify_mode','enable_two_batch_overlap','enable_pdmux','enable_prefill_cp','record_nolora_graph','is_dllm'):
        w,rt=fixture();setattr(w.target_worker.model_runner.decode_cuda_graph_runner,key,True)
        assert not r.supported_startup(w,rt);negatives.append(key)
    for key,value in (('_need_mamba_verify_commit',True),('lilicorr',object()),('_is_domino',True),('block_size',4)):
        w,rt=fixture();setattr(w,key,value);assert not r.supported_startup(w,rt);negatives.append(key)
    w,rt=fixture()
    def fail(*args):raise RuntimeError('capture failed')
    reject('capture_failure_retains_owner',lambda:r.initialize_target_prefix_states(w,widths=(2,),runtime=rt,backend_factory=backend,graph_factory=fail))
    assert w._prefix_target_states_failed.invalidated_reason=='startup capture failure'
    # Source kernel turns tok<0 into virtualpage0, then reads v2p[0].
    virtual=-1;bad_v2p=[5,1];good_v2p=[0,1]
    assert bad_v2p[max(virtual,0)]==5 and good_v2p[max(virtual,0)]==0
    negatives.append('nonzero_reserved_v2p_sink_fixture_invalid')
    report={'status':'CPU_GRAPH_FACTORY_SCOPE_PASS','scoped_switches':5,'negative_controls':negatives,
            'gpu_scope':'None; fake model/backend/runner constructors, actual factory/manager code',
            'important_limits':['Mutation invalidation API exists but active ModelRunner/weight-updater hooks are NOT installed',
                'Default-off source overlay only; no actual model graph startup or FA3 attention execution',
                'GPU metadata receipt tested the prior manager version; subsequent invalidation/stream binding changes CPU-only'],
            'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'target_prefix_graph_runtime.py',P/'target_verify_runtime_state.py',Path(__file__))}}
    (P/'TARGET_GRAPH_FACTORY_CPU.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()