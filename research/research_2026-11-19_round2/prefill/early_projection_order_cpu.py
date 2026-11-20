"""Execute scheduler startup prefix from both sources with instrumented lifetimes."""
import ast, hashlib, json, types
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
relative=Path('python/sglang/srt/managers/scheduler.py')


def execute(source,overlap,draft_present,post_active):
    module=ast.parse(source)
    cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='Scheduler')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='init_model_worker')
    stop=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.ImportFrom))
    fn.body=fn.body[:stop];fn.decorator_list=[]
    code=compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'scheduler-prefix','exec')
    events=[];resident=[False];active=['initial'];streams=[];registered=[None]
    class Context:
        def __init__(self,s):self.s=s
        def __enter__(self):self.previous=active[0];active[0]=self.s
        def __exit__(self,*args):active[0]=self.previous
    def stream_new(priority=0):
        value='schedule'+str(len(streams));streams.append(value);return value
    def initialize():
        if registered[0] is not None:return
        registered[0]=active[0]
        events.append(dict(event='projection_qualification',normal_arena_resident=resident[0],stream=active[0]))
    def graphs():resident[0]=True;events.append(dict(event='graphs_and_normal_qualification'))
    def sampling():
        initialize()
        events.append(dict(event='sampling',stream=active[0]))
    runner=types.SimpleNamespace(device='cuda',forward_stream='forward',
        token_to_kv_pool=types.SimpleNamespace(post_capture_active=post_active),
        prewarm_sampling=lambda:events.append(dict(event='target_sampling',stream=active[0])),
        post_capture_resize_kv_pool=lambda **kw:events.append(dict(event='final_kv_resize')))
    draft=types.SimpleNamespace(_initialize_projection_adapters=initialize,prewarm_sampling=sampling,
        hicache_draft_plan=None,_draft_model_runners=lambda:()) if draft_present else None
    self=types.SimpleNamespace(enable_unified_memory=False,enable_overlap=overlap,
        tp_worker=types.SimpleNamespace(model_runner=runner),draft_worker=draft,
        init_tp_model_worker=lambda:None,maybe_init_draft_worker=lambda:None,
        init_memory_pools=lambda:events.append(dict(event='initial_pools')),
        init_all_attention_backends=lambda:None,init_all_cuda_graphs=graphs)
    device=types.SimpleNamespace(Stream=stream_new,stream=Context,synchronize=lambda:events.append(dict(event='sync')))
    env=dict(torch=types.SimpleNamespace(get_device_module=lambda d:device),
        get_model=lambda:types.SimpleNamespace(is_startup_weight_load_overlap=False),
        get_parallel=lambda:types.SimpleNamespace(pp_size=1),use_mlx=lambda:False,
        time=types.SimpleNamespace(perf_counter=lambda:0),prewarm_graph_pool_borrow=lambda:None,
        kv_cache_builder=types.SimpleNamespace(prepare_hicache_staging=lambda **kw:None))
    exec(code,env);env['init_model_worker'](self)
    expected='forward' if overlap else streams[0]
    if draft_present:
        assert registered[0]==expected
        assert next(e for e in events if e['event']=='sampling')['stream']==expected
        assert len([e for e in events if e['event']=='projection_qualification'])==1
    assert len(streams)==1
    return dict(overlap=overlap,draft=draft_present,post_capture_resize=post_active,events=events)

original=(ROOT/'sglang'/relative).read_text()
changed=(HERE/'early_projection_unapplied/sources'/relative).read_text()
cases=[]
for overlap in (False,True):
 for draft in (False,True):
  for post in (False,True):
   before=execute(original,overlap,draft,post)
   after=execute(changed,overlap,draft,post)
   if draft:
    assert next(e for e in before['events'] if e['event']=='projection_qualification')['normal_arena_resident']
    assert not next(e for e in after['events'] if e['event']=='projection_qualification')['normal_arena_resident']
   cases.append(dict(before=before,after=after))
report=dict(passed=True,scope='Actual scheduler AST prefix with mocked allocations/streams. Proves host order and exact serving-stream reuse, not GPU lifetime or full-server validity.',
    source_sha256=hashlib.sha256(changed.encode()).hexdigest(),cases=cases)
(HERE/'early_projection_unapplied/order_controls.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(passed=True,cases=len(cases),source_sha256=report['source_sha256'])))
