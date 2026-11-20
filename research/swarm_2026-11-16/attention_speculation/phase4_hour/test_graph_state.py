"""CPU contract tests with synthetic qualifications, never deployment proof."""
import dataclasses,hashlib,json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from prefix_graph_state import *
P=Path(__file__).resolve().parent
calls=[];config={'width':8};draft=NS(block_size=8,graph=object())
runner=NS(attn_backend=object(),decode_cuda_graph_runner=NS(captured_req_width=8),init_new_workspace=False,model=object())
def backend(**kwargs):
    assert kwargs=={'init_new_workspace':True};calls.append(('backend',config['width']));runner.init_new_workspace=True;return object()
runner._get_attention_backend=backend
@contextmanager
def scope(width):
    old=config['width'];config['width']=width
    try:yield
    finally:config['width']=old
def factory(model_runner,**kwargs):
    assert model_runner is runner and config['width']==kwargs['speculative_num_draft_tokens']
    calls.append(('capture',config['width']));return NS(captured_req_width=config['width'])
base=(runner.attn_backend,runner.decode_cuda_graph_runner,runner.model,draft.graph)
registry=TargetGraphRegistry(runner)
state=build_target_state(runner,factory,4,scope)
assert config['width']==8 and runner.init_new_workspace is False
assert (runner.attn_backend,runner.decode_cuda_graph_runner,runner.model,draft.graph)==base
registry.register(state)
assert registry.choose(4).width==8 # Actual current native failures cannot enable prefix.
# Synthetic qualification only exercises the lifecycle mechanism.
qualified=dataclasses.replace(state,qualification=REQUIRED_GATES)
registry.states[4]=qualified
negative=[]
for width in (4,8,4,8,4):
    with registry.activate(width) as lease:
        assert lease.state.width==width and runner.decode_cuda_graph_runner.captured_req_width==width
        assert draft.block_size==8 and draft.graph is base[3]
        lease.mark_target_complete()
        try:
            with registry.activate(8):pass
        except RuntimeError:negative.append('concurrent_or_nested')
        else:raise AssertionError('lease guard')
        lease.consume_aux()
    assert (runner.attn_backend,runner.decode_cuda_graph_runner)==base[:2]
    try:lease.mark_target_complete()
    except RuntimeError:negative.append('stale_lease')
    else:raise AssertionError('stale lease')
try:
    with registry.activate(4) as lease:lease.mark_target_complete()
except RuntimeError:negative.append('unconsumed_aux')
else:raise AssertionError('aux guard')
assert registry.active is None and (runner.attn_backend,runner.decode_cuda_graph_runner)==base[:2]
def fail(*args,**kwargs):raise RuntimeError('synthetic capture failure')
try:build_target_state(runner,fail,2,scope)
except RuntimeError:negative.append('capture_failure_restoration')
else:raise AssertionError('capture should fail')
assert config['width']==8 and runner.init_new_workspace is False
assert (runner.attn_backend,runner.decode_cuda_graph_runner)==base[:2]
# Every missing qualification must retain full8 fallback.
for gate in REQUIRED_GATES:
    registry.states[4]=dataclasses.replace(state,qualification=REQUIRED_GATES-{gate})
    assert registry.choose(4).width==8
report={'status':'PASS','activation_transitions':5,'negative_cases':negative,'missing_qualification_guards':len(REQUIRED_GATES),
        'draft_and_model_identity_preserved':True,'scope':'CPU lifecycle only; all prefix widths remain unqualified in real artifacts.',
        'sources':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (P/'prefix_graph_state.py',Path(__file__))}}
(P/'cpu_graph_contract_results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
