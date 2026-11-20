"""Execute source-extracted model/worker hooks with CPU tensors and fake KV sink."""
import ast,contextlib,hashlib,inspect,json,logging,os,subprocess,sys,types
from pathlib import Path
import torch
from typing import Optional
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def method(path,name,extra=None):
    tree=ast.parse(path.read_text())
    node=next(n for cls in tree.body if isinstance(cls,ast.ClassDef) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[]
    scope=dict(torch=torch,Optional=Optional,draft_tp_context=lambda _:contextlib.nullcontext(),logger=logging.getLogger('hook-test'))
    scope.update(extra or {})
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope[name]

def main():
    torch.set_num_threads(1)
    sources=HERE/'integration_joint/sources'
    model_path=sources/'python/sglang/srt/models/dflash.py'
    worker_path=sources/'python/sglang/srt/speculative/dflash_worker_v2.py'
    project=method(model_path,'project_target_hidden')
    append=method(worker_path,'_append_target_hidden_to_draft_kv_by_loc')
    linear=torch.nn.Linear(8,4,bias=False);norm=lambda x:x*2+3
    model=types.SimpleNamespace(fc=linear,quantizable=False,hidden_norm=norm,num_context_features=2,config=types.SimpleNamespace(hidden_size=4))
    model.project_target_hidden=types.MethodType(project,model)
    worker=types.SimpleNamespace(draft_model=model,model_runner=types.SimpleNamespace(device=torch.device('cpu')),
        draft_parallel_layout=None,lilicorr=None,_use_fused_kv_materialize=False,_fused_kv_helper=None)
    calls=[];writes=[]
    def row(x):calls.append(('row',tuple(x.shape)));return linear(x)
    def col(x):calls.append(('n',tuple(x.shape)));return linear(x)
    worker._prefill_row_projection=row;worker._decode_n_projection=col
    worker._append_target_hidden_sequential=lambda **kwargs:writes.append(kwargs)
    worker._append_target_hidden_fused=lambda **kwargs:writes.append(kwargs)
    x=torch.randn(6,8);loc=torch.tensor([9,2,8,1]);pos=torch.tensor([1001,1002,65535,65536])
    for prefill in (True,False):
        append(worker,target_hidden=x,cache_loc=loc,positions=pos,prefill_projection=prefill)
        assert calls[-1]==('row' if prefill else 'n',(4,8))
        assert torch.equal(writes[-1]['ctx_hidden'],norm(linear(x[:4])))
        assert torch.equal(writes[-1]['ctx_cache_loc'],loc) and torch.equal(writes[-1]['ctx_positions'],pos)
    worker._prefill_row_projection=None;before=len(calls)
    append(worker,target_hidden=x[:4],cache_loc=loc,positions=pos,prefill_projection=True)
    assert len(calls)==before and torch.equal(writes[-1]['ctx_hidden'],norm(linear(x[:4])))
    worker._use_fused_kv_materialize=True;worker._fused_kv_helper=object()
    commit=torch.tensor([1,2],dtype=torch.int32);loc2d=loc.reshape(2,2)
    append(worker,target_hidden=x[:4],cache_loc=loc,positions=pos,cache_loc_2d=loc2d,commit_lens=commit)
    assert torch.equal(writes[-1]['ctx_cache_loc_2d'],loc2d) and torch.equal(writes[-1]['commit_lens'],commit)
    try:model.project_target_hidden(torch.randn(4,7),projection=row)
    except ValueError:pass
    else:raise AssertionError('feature validation removed')
    before=len(calls)
    append(worker,target_hidden=x[:0],cache_loc=loc[:0],positions=pos[:0],prefill_projection=True)
    assert len(calls)==before
    tree=ast.parse(worker_path.read_text())
    callsites=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_append_target_hidden_to_draft_kv_by_loc']
    assert len(callsites)==2
    assert sum(any(k.arg=='prefill_projection' and isinstance(k.value,ast.Constant) and k.value.value is True for k in n.keywords) for n in callsites)==1
    scheduler=(ROOT/'sglang/python/sglang/srt/managers/scheduler.py').read_text()
    assert scheduler.index('        self.init_memory_pools()')<scheduler.index('self.draft_worker.prewarm_sampling()')<scheduler.index('model_runner.post_capture_resize_kv_pool(')
    # Execute the startup method with fake factories: a subclass whose original
    # project_target_hidden lacks the new keyword must never receive N routing.
    factory_calls=[]
    fake_row=types.ModuleType('sglang.srt.speculative.dflash_projection_row')
    fake_n=types.ModuleType('sglang.srt.speculative.dflash_projection_n')
    def row_factory(m,*args,**kwargs):
        return object() if 'projection' in inspect.signature(m.project_target_hidden).parameters else None
    def n_factory(*args,**kwargs):factory_calls.append(kwargs);return object()
    fake_row.create_prefill_projection=row_factory;fake_n.create_projection=n_factory
    saved_modules={name:sys.modules.get(name) for name in (fake_row.__name__,fake_n.__name__)}
    sys.modules[fake_row.__name__]=fake_row;sys.modules[fake_n.__name__]=fake_n
    group=types.SimpleNamespace(world_size=8,cpu_group=None)
    parallel=types.SimpleNamespace(attn_dp_enabled=False,attn_cp_size=1,tp_group=group)
    config=types.SimpleNamespace(prefill=types.SimpleNamespace(bs=[512]),decode=types.SimpleNamespace(bs=[1,2],max_bs=2))
    init=method(worker_path,'_initialize_projection_adapters',dict(os=os,get_parallel=lambda:parallel,
                get_exec=lambda:types.SimpleNamespace(graph=types.SimpleNamespace(cuda_graph_config=config)),
                get_schedule=lambda:types.SimpleNamespace(chunked_prefill_size=512)))
    original_gather=torch.distributed.all_gather_object;original_env=os.environ.get('SGLANG_EXPERIMENTAL_DFLASH_OUTPUT_N')
    def gather(out,value,**kwargs):out[:]=[value]*len(out)
    torch.distributed.all_gather_object=gather;os.environ['SGLANG_EXPERIMENTAL_DFLASH_OUTPUT_N']='1'
    try:
        override=types.SimpleNamespace(fc=linear,project_target_hidden=lambda x:linear(x))
        negative=types.SimpleNamespace(draft_model=override,draft_parallel_layout=None,block_size=8)
        init(negative)
        assert not factory_calls and negative._prefill_row_projection is None and negative._decode_n_projection is None
        positive=types.SimpleNamespace(draft_model=model,draft_parallel_layout=None,block_size=8)
        init(positive);init(positive)
        assert len(factory_calls)==1 and factory_calls[0]['rows']==[8,16]
        assert positive._prefill_row_projection is not None and positive._decode_n_projection is not None
    finally:
        torch.distributed.all_gather_object=original_gather
        if original_env is None:os.environ.pop('SGLANG_EXPERIMENTAL_DFLASH_OUTPUT_N',None)
        else:os.environ['SGLANG_EXPERIMENTAL_DFLASH_OUTPUT_N']=original_env
        for name,value in saved_modules.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value
    report=dict(complete=True,scope='CPU execution of patched source methods; actual engine startup and CUDA/NCCL integration untested',
        checks=['row/N/native branch selection','full-M original normalization retained','trailing padding trimmed before row dispatch','positions and sparse physical write IDs retained',
                'prefix-valid commits retained','feature dimension validation retained','empty-input fast return','exactly one explicit prefill caller','initial pool then startup adapter then final pool resize',
                'override without projection keyword disables both adapters','supported model enables both independently','configured decode batches map to block rows','startup initialization is idempotent'],
        files={str(p.relative_to(sources)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (model_path,worker_path)})
    (HERE/'integration_contract.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
