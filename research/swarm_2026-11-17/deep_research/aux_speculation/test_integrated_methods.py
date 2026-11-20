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
    def wait_stream(self, stream): timeline.append(('wait_stream', stream.name, self.name))
current = [FakeStream('producer')]
ctx = (contextlib.ExitStack() if a.device == 'cpu' else contextlib.nullcontext())
with ctx as stack:
    if a.device == 'cpu':
        stack.enter_context(patch.object(torch.cuda, 'Event', FakeEvent))
        stack.enter_context(patch.object(torch.cuda, 'current_stream', lambda device=None: current[0]))
    owner = h.PrefillAuxOutputOwner(129, 64, [0,2,4,6,8], a.device)
    if a.device == 'cpu':
        same_owner = h.PrefillAuxOutputOwner(8, 4, [0,1], a.device)
        before_same = len(timeline)
        for _ in range(3):
            gen = same_owner.begin_produce()
            same_owner.end_produce(gen, same_owner.buffer)
            with same_owner.consume(same_owner.buffer):
                pass
        check('same_stream_no_sync_operations', len(timeline) == before_same)
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
    capgen = owner.begin_capture()
    check('capture_lease_acquired', owner._state == 'capturing')
    try:
        owner.begin_produce()
    except RuntimeError:
        check('capture_blocks_produce', True)
    else:
        check('capture_blocks_produce', False)
    owner.end_capture(capgen, torch.cuda.current_stream(owner.buffer.device))
    check('capture_lease_released', owner._state == 'idle')
    gen = owner.begin_produce()
    owner.end_produce(gen, owner.buffer)
    try:
        owner.begin_capture()
    except RuntimeError:
        check('pending_consumer_blocks_recapture', True)
    else:
        check('pending_consumer_blocks_recapture', False)
    with owner.consume(owner.buffer): pass
    unrelated=torch.empty_like(owner.buffer)
    with owner.consume(unrelated):
        check('unrelated_output_no_lease',owner._state=='idle')
    if a.device=='cpu':
        check('producer_waited_previous_consumer', any(t==('wait_stream','consumer','producer') for t in timeline))
        check('consumer_waited_producer', any(t==('wait_stream','producer','consumer') for t in timeline))

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

capture_wrapper = methods(runner_rel, 'PrefillCudaGraphRunner', ['capture'])['capture']
class CaptureOwner:
    def __init__(self): self.calls=[]
    def begin_capture(self): self.calls.append('begin'); return 7
    def end_capture(self,generation,stream): self.calls.append(('end',generation,stream))
co=CaptureOwner()
cr=SimpleNamespace(prefill_aux_output_owner=co,model_context=contextlib.nullcontext,
    _capture=lambda:co.calls.append('capture'),stream='capture_stream')
capture_wrapper(cr)
check('actual_recapture_wrapper_order',co.calls==['begin','capture',('end',7,'capture_stream')])
cr._capture=lambda:(_ for _ in ()).throw(ValueError('capture fixture'))
try: capture_wrapper(cr)
except ValueError: pass
check('actual_recapture_exception_cleanup',co.calls[-2:]==['begin',('end',7,'capture_stream')])

# Full owner with actual execute/consumer/capture wrappers: failures must recover.
ctx = contextlib.ExitStack() if a.device == 'cpu' else contextlib.nullcontext()
with ctx as stack:
    if a.device == 'cpu':
        current[0] = FakeStream('recovery')
        stack.enter_context(patch.object(torch.cuda, 'current_stream', lambda device=None: current[0]))
    real_owner = h.PrefillAuxOutputOwner(7, 64, [0,2,4,6,8], a.device)
    rr = SimpleNamespace(prefill_aux_output_owner=real_owner)
    recovered_reads=[]
    def read_hidden(**kwargs): recovered_reads.append(kwargs['target_hidden'].clone())
    ww = SimpleNamespace(target_worker=SimpleNamespace(model_runner=SimpleNamespace(prefill_cuda_graph_runner=rr)))
    def success(*args, **kwargs):
        real_owner.buffer.fill_(13)
        return SimpleNamespace(hidden_states=real_owner.buffer)
    def producer_failure(*args, **kwargs):
        real_owner.buffer.fill_(23)
        raise ValueError('enqueued producer writes')
    def consumer_failure(**kwargs):
        _ = kwargs['target_hidden'].clone()
        raise ValueError('enqueued consumer reads')
    for reason, body in [('producer_error',producer_failure),
                         ('nonowner_result',lambda *args,**kwargs:SimpleNamespace(hidden_states=real_owner.buffer.clone()))]:
        rr._execute_with_aux_output = body
        try: runner_execute(rr, None)
        except (ValueError,RuntimeError): pass
        else: raise AssertionError('failure not raised: '+reason)
        check('full_owner_'+reason+'_idle',real_owner._state=='idle')
        rr._execute_with_aux_output = success
        out = runner_execute(rr,None)
        ww._append_target_hidden_to_draft_kv_by_loc_impl = read_hidden
        recovered=worker_consume(ww,target_hidden=out.hidden_states)
        check('full_owner_'+reason+'_recovered',real_owner._state=='idle' and bool((recovered_reads[-1]==13).all()))
    out=runner_execute(rr,None)
    ww._append_target_hidden_to_draft_kv_by_loc_impl=consumer_failure
    try: worker_consume(ww,target_hidden=out.hidden_states)
    except ValueError: pass
    check('full_owner_consumer_error_idle',real_owner._state=='idle')
    out=runner_execute(rr,None)
    ww._append_target_hidden_to_draft_kv_by_loc_impl=read_hidden
    worker_consume(ww,target_hidden=out.hidden_states)
    check('full_owner_consumer_error_recovered',bool((recovered_reads[-1]==13).all()))
    oldgen=real_owner._generation
    newgen=real_owner.begin_produce()
    real_owner.cancel_produce(oldgen)
    check('stale_cancel_keeps_current_lease',real_owner._state=='producing')
    real_owner.cancel_produce(newgen)
    cr=SimpleNamespace(prefill_aux_output_owner=real_owner,model_context=contextlib.nullcontext,
        stream=torch.cuda.current_stream(real_owner.buffer.device))
    cr._capture=producer_failure
    try: capture_wrapper(cr)
    except ValueError: pass
    check('full_owner_capture_error_idle',real_owner._state=='idle')
    cr._capture=lambda:real_owner.buffer.fill_(31)
    capture_wrapper(cr)
    out=runner_execute(rr,None)
    with real_owner.consume(out.hidden_states):
        check('full_owner_capture_error_recovered',bool((out.hidden_states==13).all()))
    for reason, operation in [('idle_consume',lambda:real_owner.consume(real_owner.buffer).__enter__()),
                              ('stale_end',lambda:real_owner.end_produce(oldgen,real_owner.buffer))]:
        try: operation()
        except RuntimeError: check(reason+'_rejected',True)
        else: check(reason+'_rejected',False)
    for reason in ('incomplete','wrong_order','wrong_shape','wrong_dtype'):
        cap=h.DirectAuxCapture(real_owner.buffer,64,[0,2,4,6,8])
        residual=torch.zeros((7,64),dtype=torch.float32,device=a.device)
        try:
            if reason=='incomplete': cap.finalize()
            elif reason=='wrong_order': cap.capture(2,residual)
            elif reason=='wrong_shape': cap.capture(0,residual[:3])
            else: cap.capture(0,residual.to(torch.bfloat16))
        except RuntimeError: check('capture_'+reason+'_rejected',True)
        else: check('capture_'+reason+'_rejected',False)

# Worker failures between target production and append consumption release only
# the fresh generation produced by this invocation; prior pending data stays held.
outer_worker=methods(worker_rel,'DFlashWorkerV2',['forward_batch_generation'])['forward_batch_generation']
ctx=contextlib.ExitStack() if a.device=='cpu' else contextlib.nullcontext()
with ctx as stack:
    if a.device=='cpu':stack.enter_context(patch.object(torch.cuda,'current_stream',lambda device=None:current[0]))
    pending=h.PrefillAuxOutputOwner(7,64,[0,2,4,6,8],a.device)
    ww=SimpleNamespace(target_worker=SimpleNamespace(model_runner=SimpleNamespace(prefill_cuda_graph_runner=SimpleNamespace(prefill_aux_output_owner=pending))))
    def fail_before_append(*args,**kwargs):
        gen=pending.begin_produce();pending.buffer.fill_(57);pending.end_produce(gen,pending.buffer)
        raise ValueError('fixture failure between target prefill and append')
    ww._forward_batch_generation_with_aux_output=fail_before_append
    try:outer_worker(ww,None)
    except ValueError:pass
    check('worker_preconsume_error_releases_fresh_lease',pending._state=='idle')
    gen=pending.begin_produce();pending.buffer.fill_(59);pending.end_produce(gen,pending.buffer)
    with pending.consume(pending.buffer):check('worker_preconsume_error_recovers',bool((pending.buffer==59).all()))
    gen=pending.begin_produce();pending.end_produce(gen,pending.buffer)
    ww._forward_batch_generation_with_aux_output=lambda *args,**kwargs:(_ for _ in ()).throw(ValueError('before producing'))
    try:outer_worker(ww,None)
    except ValueError:pass
    check('worker_error_preserves_prior_ready_generation',pending._state=='ready' and pending.generation==gen)
    pending.discard_ready(gen-1)
    check('discard_stale_generation_preserves_ready',pending._state=='ready')
    with pending.consume(pending.buffer):pass
    ww._forward_batch_generation_with_aux_output=lambda *args,**kwargs:'result'
    check('worker_outer_preserves_return',outer_worker(ww,None)=='result' and pending._state=='idle')

# Execute the real worker prefill branch up to/through append. This directly
# reaches callback, metadata, position and consumer error sites in source.
position_failure=[False]
def position_fixture(*args):
    if position_failure[0]:raise ValueError('position fixture')
    return torch.arange(7,device=a.device),None
prefill_body=methods(worker_rel,'DFlashWorkerV2',['_forward_batch_generation_with_aux_output'],env={
    'CaptureHiddenMode':SimpleNamespace(FULL='full'),'SpecTpSyncSite':SimpleNamespace(DFLASH_TARGET='target'),
    'async_h2d':lambda values,**kwargs:torch.tensor(values,**kwargs),'compute_position':position_fixture,
})['_forward_batch_generation_with_aux_output']
ctx=contextlib.ExitStack() if a.device=='cpu' else contextlib.nullcontext()
with ctx as stack:
    if a.device=='cpu':stack.enter_context(patch.object(torch.cuda,'current_stream',lambda device=None:current[0]))
    owner=h.PrefillAuxOutputOwner(7,64,[0,2,4,6,8],a.device)
    runner=SimpleNamespace(prefill_aux_output_owner=owner)
    def target_prefill(*args,**kwargs):
        token=owner.begin_produce();owner.buffer.fill_(61);owner.end_produce(token,owner.buffer)
        return SimpleNamespace(kv_loc_plan=torch.arange(7,device=a.device),logits_output=SimpleNamespace(hidden_states=owner.buffer),next_token_ids=torch.tensor([3],device=a.device))
    target=SimpleNamespace(model_runner=SimpleNamespace(prefill_cuda_graph_runner=runner),forward_batch_generation=target_prefill)
    worker=SimpleNamespace(target_worker=target,model_runner=SimpleNamespace(prefill_attention_backend_str='fixture'),
        _validate_phase1_sampling_support=lambda batch:None,_wait_for_fused_hicache_load=lambda batch:None,
        _tp_sync=SimpleNamespace(sync=lambda *args:None),_draft_write_ids=lambda plan:plan,
        _make_next_draft_input_prefill=lambda **kwargs:'next')
    worker._forward_batch_generation_with_aux_output=lambda *args,**kwargs:prefill_body(worker,*args,**kwargs)
    worker._append_target_hidden_to_draft_kv_by_loc=lambda **kwargs:worker_consume(worker,**kwargs)
    reads=[]
    worker._append_target_hidden_to_draft_kv_by_loc_impl=lambda **kwargs:reads.append(kwargs['target_hidden'].clone())
    for reason in ['publish','extend_metadata','cache_metadata','position']:
        batch=SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda:True,is_idle=lambda:False),is_extend_in_batch=False,
            seq_lens=torch.tensor([7],device=a.device),extend_lens=[7],prefix_lens=[0],out_cache_loc=torch.arange(7,device=a.device))
        publish=None
        if reason=='publish':publish=lambda *args:(_ for _ in ()).throw(ValueError('publish fixture'))
        if reason=='extend_metadata':batch.extend_lens=None
        if reason=='cache_metadata':batch.out_cache_loc=None
        position_failure[0]=reason=='position'
        try:outer_worker(worker,batch,on_publish=publish)
        except (ValueError,RuntimeError):pass
        else:raise AssertionError('real prefill failure did not raise: '+reason)
        check('real_prefill_'+reason+'_releases',owner._state=='idle')
        batch.extend_lens=[7];batch.out_cache_loc=torch.arange(7,device=a.device);position_failure[0]=False
        result=outer_worker(worker,batch)
        check('real_prefill_'+reason+'_recovers',owner._state=='idle' and result.logits_output.hidden_states is None and result.next_draft_input=='next' and bool((reads[-1]==61).all()))

if a.device!='cpu': torch.cuda.synchronize()
report={'status':'PASS','device':a.device,'checks':checks,'check_count':len(checks),
    'scope':'Actual edited methods extracted unchanged; model layers, runtime/distributed environment, and CPU stream timeline are fixtures. No full SGLang model execution.',
    'source_manifest_sha256':hashlib.sha256((P/'source_manifest.json').read_bytes()).hexdigest(),
    'stream_timeline':timeline if a.device=='cpu' else None}
a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':len(checks),'device':a.device}))
