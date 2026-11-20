"""CUDA owner races, graph replay and isolated allocator study.

Runs only under root GPU lease. Uses exact edited model/BCG methods with toy
layers; full graph captures simulate transformer-body capture, not segmented
SGLang BCG runtime or attention/model kernels.
"""
import argparse
import ast
import contextlib
import copy
import gc
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace
import torch

P=Path(__file__).resolve().parent
pa=argparse.ArgumentParser()
pa.add_argument('--mode', choices=['race','baseline','candidate'], required=True)
pa.add_argument('--out',type=Path,required=True)
a=pa.parse_args()
checks=[]
def check(name,ok,**kw):
    checks.append(dict(name=name,passed=bool(ok),**kw))
    if not ok: raise AssertionError(name)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
h=load('aux_owner',P/'candidate_sources/python/sglang/srt/layers/prefill_aux_output.py')

def method(rel,cls,name,baseline=False,env=None):
    path=P/('base_sources' if baseline else 'candidate_sources')/rel
    tree=ast.parse(path.read_text()); cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    fn=copy.deepcopy(next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name==name))
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[]);ast.fix_missing_locations(mod)
    ns={'torch':torch,**(env or {})};exec(compile(mod,str(path),'exec'),ns);return ns[name]

report={'mode':a.mode,'device':torch.cuda.get_device_name(),'started':time.time(),'scope':'Extracted real edited methods; fixture layers and single CUDAGraph body, not SGLang BCG runtime/full model.'}
if a.mode=='race':
    # Allocation at prior illustrative actual dimensions, then release it.
    before=torch.cuda.memory_allocated()
    capacity_owner=h.PrefillAuxOutputOwner(16384,8192,[0,1,2,3,4],'cuda')
    report['capacity_allocation']={'rows':16384,'hidden':8192,'captures':5,'bytes':torch.cuda.memory_allocated()-before,'note':'Capacity illustration only: exact checkpoint width/IDs not independently verified.'}
    del capacity_owner;gc.collect();torch.cuda.empty_cache()
    owner=h.PrefillAuxOutputOwner(256,128,[0,1,2,3,4],'cuda')
    producer=torch.cuda.Stream();consumer=torch.cuda.Stream();host=torch.cuda.current_stream()
    producer.wait_stream(host);consumer.wait_stream(host)
    observed=[];expected=[]
    for gen in range(12):
        rows=[256,17,129,1][gen%4]
        with torch.cuda.stream(producer):
            token=owner.begin_produce();owner.view(rows).fill_(gen+1);owner.end_produce(token,owner.view(rows))
        with torch.cuda.stream(consumer):
            with owner.consume(owner.view(rows)):
                torch.cuda._sleep(3000000)
                observed.append(owner.view(rows).clone())
                expected.append(gen+1)
    host.wait_stream(producer);host.wait_stream(consumer);torch.cuda.synchronize()
    for i,(x,value) in enumerate(zip(observed,expected)):
        check('cross_stream_generation_%d'%i,bool((x==value).all().item()))
    # Negative control deliberately removes the wait for prior reads. Consumer
    # sleep makes overwrite reproducible; retain any failure to induce a race.
    bad=h.PrefillAuxOutputOwner(256,128,[0,1,2,3,4],'cuda')
    with torch.cuda.stream(producer):
        token=bad.begin_produce();bad.buffer.fill_(3);bad.end_produce(token,bad.buffer)
    with torch.cuda.stream(consumer):
        with bad.consume(bad.buffer):
            torch.cuda._sleep(60000000);negative=bad.buffer.clone()
    with torch.cuda.stream(producer):
        bad._have_consumed=False
        token=bad.begin_produce();bad.buffer.fill_(9);bad.end_produce(token,bad.buffer)
    torch.cuda.synchronize()
    check('negative_control_missing_wait_corrupts',not torch.equal(negative,torch.full_like(negative,3)))
    report['negative_control_values']=negative.unique().cpu().tolist()
    # Error paths and generation checks on real CUDA events.
    with bad.consume(bad.buffer): pass
    token=bad.begin_produce()
    try: bad.end_produce(token-1,bad.buffer)
    except RuntimeError: check('stale_generation_rejected',True)
    else: check('stale_generation_rejected',False)
    bad.cancel_produce(token)
    token=bad.begin_produce();bad.buffer.fill_(11);bad.end_produce(token,bad.buffer)
    with bad.consume(torch.empty_like(bad.buffer)):
        check('nonowner_leaves_ready_lease',bad._state=='ready')
    try:
        with bad.consume(bad.buffer):
            _=bad.buffer.clone();raise ValueError('intentional consumer error')
    except ValueError: pass
    check('consumer_exception_releases',bad._state=='idle')
    token=bad.begin_produce();bad.buffer.zero_();bad.cancel_produce(token)
    torch.cuda.synchronize()
else:
    baseline=a.mode=='baseline'
    env={'direct_aux_capture':h.direct_aux_capture,'get_forward':lambda:SimpleNamespace(set=lambda *args:None),
         'layernorm_sp':SimpleNamespace(runs_sp=lambda mode:False)}
    forward=method('python/sglang/srt/models/muse_spark.py','MuseSpark1xModel','forward',baseline,env)
    class Layer:
        def __init__(self,i):self.i=i;self.self_attn_input_norm=lambda x:x
        def __call__(self,positions,stream,attn_input,fb):
            stream.add_((self.i+1)*0.03125);stream.mul_(1.015625);return stream,stream
    hidden=1024;maxrows=16384;ids=[0,2,4,6,8];buckets=[16384,4096,1024,256]
    model=SimpleNamespace(embed_norm=None,layers=[Layer(i) for i in range(9)],layers_to_capture=ids,
        residual_dtype=torch.float32,operator_dtype=torch.bfloat16,norm=lambda x:x)
    owner=None if baseline else h.PrefillAuxOutputOwner(maxrows,hidden,ids,'cuda')
    class Backend:pass
    for name in ('_alloc_full_buffer','_copy_output_to_buffer','_slice_output'):
        setattr(Backend,name,method('python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py','BreakableCudaGraphBackend',name,baseline,{'PPProxyTensors':type('PPProxyTensors',(),{})}))
    backend=Backend();backend._cuda_graph_runner=SimpleNamespace(adopt_owned_cuda_graph_output=(owner.adopt if owner else lambda *args:None))
    inputs=torch.zeros((maxrows,hidden),device='cuda',dtype=torch.bfloat16)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());graphs={};outputs={};pool=torch.cuda.graph_pool_handle();shared=None
    torch.cuda.reset_peak_memory_stats()
    for rows in buckets:
        fb=SimpleNamespace(aux_hidden_states_buffer=owner.view(rows) if owner else None,forward_mode=SimpleNamespace(is_extend=lambda:True))
        with torch.cuda.stream(stream):
            for _ in range(2):warm=forward(model,None,None,fb,inputs[:rows])
            if shared is None:shared=backend._alloc_full_buffer(warm,maxrows)
            del warm
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=pool,stream=stream):
                result=forward(model,None,None,fb,inputs[:rows]);backend._copy_output_to_buffer(result,shared,rows)
            del result
        graphs[rows]=graph;outputs[rows]=backend._slice_output(shared,rows)
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();gc.collect()
    report['after_capture']={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'max_allocated':torch.cuda.max_memory_allocated(),'max_reserved':torch.cuda.max_memory_reserved()}
    for cycle,rows in enumerate([256,16384,1024,4096,256,16384]):
        value=cycle*0.25
        if owner:token=owner.begin_produce()
        inputs[:rows].fill_(value);graphs[rows].replay()
        out=outputs[rows][1];packed=torch.cat(out,dim=-1) if baseline else out
        if owner:owner.end_produce(token,packed)
        with owner.consume(packed) if owner else contextlib.nullcontext():
            snapshot=packed.clone()
        # Independent scalar recurrence with the same FP32 ops and casts.
        v=torch.tensor(value,dtype=torch.float32,device='cuda');expect=[]
        for i in range(9):
            v=(v+(i+1)*0.03125)*1.015625
            if i in ids:expect.append(v.to(torch.bfloat16))
        for tap,val in enumerate(expect):check('graph_cycle_%d_tap_%d'%(cycle,tap),bool((snapshot[:,tap*hidden:(tap+1)*hidden]==val).all().item()))
        del packed,snapshot
    torch.cuda.synchronize();gc.collect()
    report['after_replay']={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'max_allocated':torch.cuda.max_memory_allocated(),'max_reserved':torch.cuda.max_memory_reserved()}
    report['geometry']={'max_rows':maxrows,'hidden':hidden,'capture_ids':ids,'buckets':buckets,'note':'Fixture captures and width; max row capacity matches configured prefill size.'}
report.update(status='PASS',checks=checks,check_count=len(checks),finished=time.time())
a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'PASS','mode':a.mode,'checks':len(checks)}))
