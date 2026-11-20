"""Execute real edited methods with tensor fixtures; no model weights required.

These are integration-contract checks, not a full Muse/SGLang serving test.
Heavy model/distributed imports are replaced, while edited methods are extracted
unchanged by AST from the source overlay and paired baseline snapshots.
"""
import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

P = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu')
parser.add_argument('--out', type=Path, default=P / 'cpu_integration_results.json')
a = parser.parse_args()
checks = []


def check(name, ok, **extra):
    checks.append(dict(name=name, passed=bool(ok), **extra))
    if not ok:
        raise AssertionError(name)


def load_helper():
    path = P / 'candidate_sources/python/sglang/srt/layers/prefill_aux_output.py'
    spec = importlib.util.spec_from_file_location('aux_output_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h = load_helper()


def methods(rel, clsname, names, base=False, env=None):
    path = P / ('base_sources' if base else 'candidate_sources') / rel
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == clsname)
    nodes = [copy.deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names), (clsname, names)
    mod = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    ast.fix_missing_locations(mod)
    scope = {'torch': torch, **(env or {})}
    exec(compile(mod, str(path), 'exec'), scope)
    return {name: scope[name] for name in names}


class Mode:
    def __init__(self, extend=True):
        self.extend = extend
    def is_extend(self):
        return self.extend


sp = SimpleNamespace(active=False)
layernorm_sp = SimpleNamespace(runs_sp=lambda mode: sp.active, layernorm_sp_enabled=lambda: sp.active)
env = dict(direct_aux_capture=h.direct_aux_capture, layernorm_sp=layernorm_sp,
           get_forward=lambda: SimpleNamespace(set=lambda *args: None))
model_rel = 'python/sglang/srt/models/muse_spark.py'
model_new = methods(model_rel, 'MuseSpark1xModel', ['forward', 'prefill_aux_output_spec'], env=env)
model_old = methods(model_rel, 'MuseSpark1xModel', ['forward'], base=True, env=env)
wrapper_spec = methods(model_rel, 'MuseSparkModel', ['prefill_aux_output_spec'])


class Layer:
    def __init__(self, index):
        self.index = index
        self.self_attn_input_norm = lambda x: x
    def __call__(self, positions, stream, attn_input, fb):
        # Reuse residual storage to detect lost snapshots.
        stream.add_((self.index + 1) * 0.03125)
        stream.mul_(1.015625)
        return stream, stream


def fixture(rows, hidden=64):
    layers = [Layer(i) for i in range(9)]
    obj = SimpleNamespace(layers=layers, layers_to_capture=[0, 2, 4, 6, 8],
        operator_dtype=torch.bfloat16, residual_dtype=torch.float32,
        embed_norm=None, norm=lambda x: x, config=SimpleNamespace(hidden_size=hidden))
    obj.prefill_aux_output_spec = lambda: model_new['prefill_aux_output_spec'](obj)
    inputs = torch.randn((rows, hidden), generator=torch.Generator().manual_seed(rows), dtype=torch.float32).to(a.device).to(torch.bfloat16)
    fb = SimpleNamespace(aux_hidden_states_buffer=None, forward_mode=Mode())
    return obj, inputs, fb


for rows in (1, 3, 7, 33, 129):
    obj, x, fb = fixture(rows)
    ref_hidden, ref_aux = model_old['forward'](obj, None, None, fb, x.clone())
    buffer = torch.empty((rows, 320), device=a.device, dtype=torch.bfloat16)
    fb.aux_hidden_states_buffer = buffer
    hidden, aux = model_new['forward'](obj, None, None, fb, x.clone())
    check('model_hidden_%s' % rows, torch.equal(hidden, ref_hidden))
    check('model_aux_%s' % rows, torch.equal(aux, torch.cat(ref_aux, dim=-1)))
    check('model_owner_%s' % rows, aux.data_ptr() == buffer.data_ptr())
    for reason in ('decode', 'wrong_width', 'wrong_dtype', 'unsorted_ids', 'sp'):
        fb.forward_mode = Mode(reason != 'decode')
        fb.aux_hidden_states_buffer = buffer
        if reason == 'wrong_width': fb.aux_hidden_states_buffer = buffer[:, :-1]
        if reason == 'wrong_dtype': fb.aux_hidden_states_buffer = buffer.float()
        if reason == 'unsorted_ids': obj.layers_to_capture = [8, 0, 4]
        if reason == 'sp':
            # SP entry/exit isn't a fixture dependency; guard is tested directly.
            result = h.direct_aux_capture(buffer, x.float(), [0,2,4,6,8], torch.bfloat16, True)
            check('sp_fallback_%s' % rows, result is None)
            continue
        rh, ra = model_old['forward'](obj, None, None, fb, x.clone())
        nh, na = model_new['forward'](obj, None, None, fb, x.clone())
        check('fallback_%s_%s' % (reason, rows), isinstance(na, list) and torch.equal(nh,rh) and all(torch.equal(u,v) for u,v in zip(na,ra)))
        obj.layers_to_capture = [0,2,4,6,8]

# Conversion edge cases compare bit patterns, including signed zero and NaN.
values = torch.tensor([0.0, -0.0, float('inf'), -float('inf'), float('nan'),
    1.00390625, -1.00390625, 1e-40, -1e-40, 3.38953139e38],
    dtype=torch.float32, device=a.device).view(2,5)
packed = torch.empty((2,10), dtype=torch.bfloat16, device=a.device)
cap = h.direct_aux_capture(packed, values, [0,1], torch.bfloat16, False)
cap.capture(0, values); cap.capture(1, values)
expected = torch.cat([values.to(torch.bfloat16), values.to(torch.bfloat16)],dim=-1)
check('conversion_edge_bits', torch.equal(cap.finalize().view(torch.int16),expected.view(torch.int16)))

obj, x, fb = fixture(8)
wrapper = SimpleNamespace(language_model=obj)
check('wrapper_spec_reaches_inner_model', wrapper_spec['prefill_aux_output_spec'](wrapper) == (64,(0,2,4,6,8)))
for ids in ([], [0,0], [1,0], [-1], [9], [False], ['1']):
    obj.layers_to_capture = ids
    check('bad_spec_' + repr(ids), obj.prefill_aux_output_spec() is None)


# CPU stream/event model records order but does not claim real GPU synchronization.
timeline = []
class FakeEvent:
    def record(self, stream): timeline.append(('record', id(self), stream.name))
class FakeStream:
    def __init__(self, name): self.name=name
    def wait_event(self, event): timeline.append(('wait', id(event), self.name))
current = [FakeStream('producer')]
ctx = (contextlib.ExitStack() if a.device == 'cpu' else contextlib.nullcontext())
with ctx as stack:
    if a.device == 'cpu':
        stack.enter_context(patch.object(torch.cuda, 'Event', FakeEvent))
        stack.enter_context(patch.object(torch.cuda, 'current_stream', lambda device=None: current[0]))
    owner = h.PrefillAuxOutputOwner(129, 64, [0,2,4,6,8], a.device)
    backend_rel = 'python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py'
    backend_methods = methods(backend_rel, 'BreakableCudaGraphBackend',
        ['_alloc_full_buffer','_slice_output','_copy_output_to_buffer'],
        env={'PPProxyTensors':type('PPProxyTensors',(),{})})
    class Backend: pass
    for name, fn in backend_methods.items(): setattr(Backend,name,fn)
    backend=Backend()
    backend._cuda_graph_runner=SimpleNamespace(adopt_owned_cuda_graph_output=owner.adopt)
    src=(torch.randn((129,64),device=a.device), owner.view(129))
    stored=backend._alloc_full_buffer(src,129)
    check('bcg_owned_aux_adoption', stored[1].data_ptr()==owner.buffer.data_ptr())
    check('bcg_regular_output_separate', stored[0].data_ptr()!=src[0].data_ptr())
    for rows in (129,33,7,1,33,129):
        obj,x,fb=fixture(rows)
        fb.aux_hidden_states_buffer=owner.view(rows)
        generation=owner.begin_produce()
        try:
            owner.begin_produce()
            check('busy_producer_rejected',False)
        except RuntimeError:
            check('busy_producer_rejected_%s'%rows,True)
        result=model_new['forward'](obj,None,None,fb,x.clone())
        backend._copy_output_to_buffer(result, stored, rows)
        sliced=backend._slice_output(stored,rows)
        owner.end_produce(generation,sliced[1])
        if a.device=='cpu': current[0]=FakeStream('consumer')
        with owner.consume(sliced[1]):
            copied=sliced[1].clone()
            try:
                owner.begin_produce()
                check('consumer_lease_blocks_overwrite',False)
            except RuntimeError:
                check('consumer_lease_blocks_overwrite_%s'%rows,True)
        if a.device=='cpu': current[0]=FakeStream('producer')
        _,ref=model_old['forward'](obj,None,None,fb,x.clone())
        check('bcg_alternating_bucket_%s'%rows,torch.equal(copied,torch.cat(ref,dim=-1)))
    aliases=[owner.buffer.clone(), owner.buffer[1:], owner.buffer.t(), owner.buffer[:,1:], owner.buffer.float()]
    for i, bad in enumerate(aliases):
        check('adoption_reject_%d'%i,owner.adopt(bad,129) is None)
    check('adoption_over_capacity',owner.adopt(owner.buffer,130) is None)
    check('adoption_under_output',owner.adopt(owner.buffer,128) is None)
    generation=owner.begin_produce()
    owner.cancel_produce(generation)
    check('producer_cancel_releases',owner._state=='idle')
    token = owner.begin_produce()
    try:
        owner.end_produce(token-1,owner.buffer)
    except RuntimeError:
        check('stale_generation_rejected',True)
    else:
        check('stale_generation_rejected',False)
    owner.cancel_produce(token)
    token = owner.begin_produce()
    owner.end_produce(token,owner.buffer)
    with owner.consume(torch.empty_like(owner.buffer)):
        check('nonowner_leaves_ready_lease',owner._state=='ready')
    try:
        with owner.consume(owner.buffer):
            raise ValueError('intentional consumer error')
    except ValueError:
        pass
    check('consumer_exception_releases',owner._state=='idle')
    unrelated=torch.empty_like(owner.buffer)
    with owner.consume(unrelated):
        check('unrelated_output_no_lease',owner._state=='idle')
    if a.device=='cpu':
        check('producer_waited_previous_consumer', any(t[0]=='wait' and t[1]==id(owner._consumed) and t[2]=='producer' for t in timeline))
        check('consumer_waited_producer', any(t[0]=='wait' and t[1]==id(owner._ready) and t[2]=='consumer' for t in timeline))

# Actual runner execute and worker consumer wrappers, with an instrumented body.
runner_rel='python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py'
runner_execute=methods(runner_rel,'PrefillCudaGraphRunner',['execute'])['execute']
worker_rel='python/sglang/srt/speculative/dflash_worker_v2.py'
worker_consume=methods(worker_rel,'DFlashWorkerV2',['_append_target_hidden_to_draft_kv_by_loc'],env={'nullcontext':contextlib.nullcontext})['_append_target_hidden_to_draft_kv_by_loc']
class DummyOwner:
    def __init__(self): self.calls=[]
    def begin_produce(self): self.calls.append('begin'); return 1
    def end_produce(self,gen,output): self.calls.append('end'); assert output=='hidden'
    def cancel_produce(self,gen): self.calls.append('cancel')
    @contextlib.contextmanager
    def consume(self,output):
        self.calls.append('consume_begin')
        yield
        self.calls.append('consume_end')
d=DummyOwner()
r=SimpleNamespace(prefill_aux_output_owner=d,_execute_with_aux_output=lambda *args,**kwargs:SimpleNamespace(hidden_states='hidden'))
runner_execute(r,None)
w=SimpleNamespace(target_worker=SimpleNamespace(model_runner=SimpleNamespace(prefill_cuda_graph_runner=r)),
    _append_target_hidden_to_draft_kv_by_loc_impl=lambda **kwargs:d.calls.append('materialize'))
worker_consume(w,target_hidden='hidden')
check('actual_wrapper_order',d.calls==['begin','end','consume_begin','materialize','consume_end'],calls=d.calls)
r._execute_with_aux_output=lambda *args,**kwargs:(_ for _ in ()).throw(ValueError('fixture'))
try: runner_execute(r,None)
except ValueError: pass
check('actual_wrapper_exception_cancel',d.calls[-2:]==['begin','cancel'])

if a.device!='cpu': torch.cuda.synchronize()
report={'status':'PASS','device':a.device,'checks':checks,'check_count':len(checks),
    'scope':'Actual edited methods extracted unchanged; model layers, runtime/distributed environment, and CPU stream timeline are fixtures. No full SGLang model execution.',
    'source_manifest_sha256':hashlib.sha256((P/'source_manifest.json').read_bytes()).hexdigest(),
    'stream_timeline':timeline if a.device=='cpu' else None}
a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':len(checks),'device':a.device}))
