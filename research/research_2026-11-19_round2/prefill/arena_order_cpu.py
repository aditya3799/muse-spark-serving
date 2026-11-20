"""Execute actual normal-package dispatch control with mocked allocation only."""
import ast
import hashlib
import json
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
SERVING=HERE.parents[1]
source_path=SERVING/'research_2026-11-19/moe/normal_package/muse_spark_grouped_gate.py'
source=source_path.read_text()
tree=ast.parse(source)
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='try_grouped')
compiled=compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(source_path),'exec')


class Tensor:
    def __init__(self,shape,dtype):
        self.shape=shape;self.dtype=dtype;self.ndim=len(shape)
        self.device=types.SimpleNamespace(index=0)
        self.is_cuda=True
    def size(self,dim):return self.shape[dim]
    def is_contiguous(self):return True


def scenario(qualify_enabled,initial_capture_only=False):
    events=[];state={};capturing=[False]
    class Arena:
        def forward(self,*args):events.append('normal_forward');return 'candidate'
    torch=types.SimpleNamespace(bfloat16='bf16',int32='int32',
        cuda=types.SimpleNamespace(is_current_stream_capturing=lambda:capturing[0],
        current_stream=lambda device:types.SimpleNamespace(cuda_stream=3)))
    def qualify(*args):
        events.append('qualify_and_retain_arena' if qualify_enabled else 'qualify_rejected')
        return Arena() if qualify_enabled else False
    g=dict(_BUILD=True,_STATE=state,MIN_M=4096,MAX_REQUEST_M=16384,torch=torch,
        _meta_ok=lambda layer:True,_serialized_execution=lambda:True,
        _stream_allowed=lambda arena:True,_qualify=qualify,
        LOG=types.SimpleNamespace(info=lambda *args:None))
    exec(compiled,g)
    x=Tensor((16384,4096),'bf16');ids=Tensor((16384,8),'int32');ids.device=x.device
    def call():return g['try_grouped'](None,object(),x,ids,torch,global_experts=256)
    if not initial_capture_only:
        events.append('prefill_warmup_before_capture');call();call()
    capturing[0]=True;events.append('prefill_capture');call();capturing[0]=False
    events.append('draft_projection_qualification_after_all_graphs')
    resident=isinstance(state.get(0),Arena)
    if qualify_enabled and not initial_capture_only:
        assert resident
    else:assert not resident
    return dict(qualify_enabled=qualify_enabled,capture_only=initial_capture_only,
                events=events,arena_resident_at_projection_qualification=resident)


report=dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    scope='Actual try_grouped control executed with mocked tensors/Arena; scheduler and BCG order source-audited. Not GPU or full-server execution.',
    cases=[scenario(True),scenario(False),scenario(True,True)],
    order_sources=[
        'scheduler.init_model_worker: init_all_cuda_graphs BEFORE draft_worker.prewarm_sampling',
        'prefill runner _capture_one_stream: largest token bucket first',
        'BreakableCudaGraphBackend.capture_one: two eager warmups BEFORE graph capture',
        'normal try_grouped: first eligible non-capturing call qualifies, successfulArena retained in global_STATE',
    ],passed=True)
(HERE/'arena_order_cpu.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
