"""CPU execution of the actual assembled scheduler initialization prefix with fake streams."""
import ast,contextlib,hashlib,itertools,json,time,types
from pathlib import Path
R=Path('D:/OpenKernelChallenge/serving');O=R/'research_2026-11-20/local_validation'
p=R/'research_2026-11-19_final/assembled/python/sglang/srt/managers/scheduler.py'
tree=ast.parse(p.read_text());node=next(n for cls in tree.body if isinstance(cls,ast.ClassDef) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='init_model_worker')
cut=next(i for i,n in enumerate(node.body) if isinstance(n,ast.ImportFrom))
assert node.body[cut].module=='sglang.srt.model_executor.model_runner_components.cuda_graph_setup'
node.body=node.body[:cut];node.decorator_list=[]
results=[]
for overlap,pp,draft_enabled,resize in itertools.product((False,True),(1,2),(False,True),(False,True)):
    events=[];current=['default'];forward=object()
    def record(name):events.append((name,current[0]))
    class Device:
        @staticmethod
        def Stream(priority):return object()
        @staticmethod
        @contextlib.contextmanager
        def stream(value):
            old=current[0];current[0]=value
            try:yield
            finally:current[0]=old
        @staticmethod
        def synchronize():record('sync')
    runner=types.SimpleNamespace(device='cuda',forward_stream=forward,token_to_kv_pool=types.SimpleNamespace(post_capture_active=resize),prewarm_sampling=lambda:record('prewarm'),post_capture_resize_kv_pool=lambda **k:record('resize'))
    worker=types.SimpleNamespace(model_runner=runner)
    draft=types.SimpleNamespace(_initialize_projection_adapters=lambda:record('adapter'),prewarm_sampling=lambda:record('prewarm'),_draft_model_runners=lambda:[],hicache_draft_plan=None) if draft_enabled else None
    obj=types.SimpleNamespace(tp_worker=worker,draft_worker=draft,enable_overlap=overlap,enable_unified_memory=False,init_tp_model_worker=lambda:None,maybe_init_draft_worker=lambda:None,init_memory_pools=lambda:record('pools'),init_all_attention_backends=lambda:record('attention'),init_all_cuda_graphs=lambda:record('graphs'))
    ns=dict(time=time,torch=types.SimpleNamespace(get_device_module=lambda _:Device),get_model=lambda:types.SimpleNamespace(is_startup_weight_load_overlap=False),get_parallel=lambda:types.SimpleNamespace(pp_size=pp),use_mlx=lambda:False,prewarm_graph_pool_borrow=lambda:record('borrow'),kv_cache_builder=types.SimpleNamespace(prepare_hicache_staging=lambda **k:record('staging')))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns);ns['init_model_worker'](obj)
    expected=forward if overlap or pp>1 else obj.schedule_stream
    names=[e[0] for e in events]
    assert names.index('pools')<names.index('graphs')<names.index('prewarm')
    assert next(stream for name,stream in events if name=='prewarm') is expected
    if draft_enabled:
        assert names.index('pools')<names.index('adapter')<names.index('graphs')
        assert next(stream for name,stream in events if name=='adapter') is expected
    if resize:assert names.index('prewarm')<names.index('resize')
    results.append(dict(overlap=overlap,pp=pp,draft=draft_enabled,resize=resize,passed=True,events=names))
doc=dict(complete=True,correct=True,scope=__doc__,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),cases=results)
(O/'scheduler_stream_contract.json').write_text(json.dumps(doc,indent=2));print('PASS scheduler initialization stream/order cases:',len(results))
