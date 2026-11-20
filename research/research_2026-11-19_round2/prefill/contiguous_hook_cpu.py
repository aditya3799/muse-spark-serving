"""Host-only admission/default/AST tests for the unapplied diagnostic hook."""
import ast
import hashlib
import importlib.util
import itertools
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('kv_helper', HERE / 'contiguous_prefill_helper.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class Poison:
    def __getattr__(self, name):
        raise AssertionError('disabled path inspected ' + name)


assert helper.ENABLED is False
assert helper.maybe_stage(*([Poison()] * 8), allowed=True) is None
assert 'torch' not in sys.modules
controls = {'disabled_default_no_tensor_access_or_torch_import': True}
for bits in itertools.product((False, True), repeat=5):
    got = helper.lifecycle_allowed(**dict(zip(
        ('enabled', 'allowed', 'capture_setup', 'full_graph', 'capturing'), bits)))
    assert got == (bits[0] and bits[1] and not any(bits[2:]))
controls['lifecycle_combinations'] = 32


class Tensor:
    def __init__(self, shape, dtype='fp8', device='cuda:0', contiguous=True):
        self.shape, self.dtype, self.device, self.ndim = shape, dtype, device, len(shape)
        self.contiguous = contiguous
    def is_contiguous(self):
        return self.contiguous
    def stride(self, dim):
        return 1 if dim == 1 else self.shape[1]


class ReachedAllocation(Exception):
    pass


def allocate(*args, **kwargs):
    raise ReachedAllocation()


torch = types.ModuleType('torch')
torch.float8_e4m3fn, torch.int32, torch.uint8 = 'fp8', 'int32', 'uint8'
torch.empty = allocate
torch.cuda = types.SimpleNamespace(
    is_current_stream_capturing=lambda: False,
    get_device_capability=lambda device: (9, 0),
    current_stream=lambda device: types.SimpleNamespace(cuda_stream=3),
    memory_allocated=lambda device: 0, memory_reserved=lambda device: 0,
    OutOfMemoryError=MemoryError,
)
sys.modules['torch'] = torch
fbmod = types.ModuleType('sglang.srt.model_executor.forward_batch_info')
fbmod.ForwardMode = types.SimpleNamespace(EXTEND='extend')
sys.modules[fbmod.__name__] = fbmod
scope = types.ModuleType('sglang.srt.model_executor.runner_utils.prefill_graph')
scope.is_in_full_prefill_graph = lambda: False
sys.modules[scope.__name__] = scope
runtime = types.ModuleType('sglang.srt.runtime_context')
flags = types.SimpleNamespace(dp=types.SimpleNamespace(capturing_prefill_graph=False))
parallel = types.SimpleNamespace(pp_size=1, enable_prefill_cp=False)
runtime.get_flags = lambda: flags
runtime.get_parallel = lambda: parallel
sys.modules[runtime.__name__] = runtime
kernel = types.ModuleType('sglang.kernels.ops.kvcache.contiguous_kv_staging')
kernel._pack = None
sys.modules[kernel.__name__] = kernel
helper.ENABLED = True


def objects():
    backend = types.SimpleNamespace(fa_impl_ver=3, page_size=1, kv_cache_is_mxfp8=False,
                                    use_mla=False, fa_skip_kv_cache=False)
    layer = types.SimpleNamespace(is_cross_attention=False, head_dim=64, v_head_dim=64,
                                  tp_k_head_num=2, tp_v_head_num=2, tp_q_head_num=16)
    fb = types.SimpleNamespace(forward_mode='extend', dcp_kv_mask=None)
    return [backend, Tensor((753, 1024)), Tensor((8192, 1, 2, 64)),
            Tensor((8192, 1, 2, 64)), Tensor((1, 8192), 'int32'),
            Tensor((1,), 'int32'), layer, fb]


# Positive control reaches allocation only after all admission checks.
args = objects()
try:
    helper.maybe_stage(*args, allowed=True)
except ReachedAllocation:
    controls['eligible_reaches_one_lazy_allocation'] = True
else:
    raise AssertionError('eligible case did not reach allocation')
negatives = []
changes = [(0, 'fa_impl_ver', 4), (0, 'page_size', 16), (0, 'kv_cache_is_mxfp8', True),
           (0, 'use_mla', True), (0, 'fa_skip_kv_cache', True),
           (6, 'is_cross_attention', True), (6, 'head_dim', 128),
           (6, 'v_head_dim', 128), (6, 'tp_v_head_num', 1),
           (6, 'tp_q_head_num', 15), (7, 'dcp_kv_mask', object()),
           (7, 'forward_mode', 'decode'), (7, 'forward_mode', 'verify'),
           (1, 'dtype', 'bf16'), (2, 'contiguous', False),
           (4, 'dtype', 'int64'), (5, 'shape', (2,)), (3, 'device', 'cuda:1')]
for index, name, value in changes:
    args = objects()
    setattr(args[index], name, value)
    assert helper.maybe_stage(*args, allowed=True) is None
    negatives.append([index, name, str(value)])
args = objects(); args[4].shape = (32, 131072); args[5].shape = (32,)
assert helper.maybe_stage(*args, allowed=True) is None
controls['oversize_full_graph_capacity_rejected'] = True
torch.cuda.get_device_capability = lambda _: (8, 6)
assert helper.maybe_stage(*objects(), allowed=True) is None
controls['sm86_rejected'] = True
controls['negative_cases'] = negatives

baseline = HERE.parents[1] / 'release_2026-11-18/candidate_sources/python/sglang/srt/layers/attention/flashattention_backend.py'
patched = HERE / 'contiguous_kv_unapplied/sources/python/sglang/srt/layers/attention/flashattention_backend.py'


def native_call(path):
    hits = []
    tree = ast.parse(path.read_text())
    body = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'forward_extend')
    for node in ast.walk(body):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'flash_attn_with_kvcache':
            if any(k.arg == 'scheduler_metadata' and isinstance(k.value, ast.Name)
                   and k.value.id == 'sched_meta' for k in node.keywords):
                hits.append(node)
    assert len(hits) == 1
    return hits[0]


old, new = native_call(baseline), native_call(patched)
oldkw = {k.arg: ast.dump(k.value) for k in old.keywords}
newkw = {k.arg: ast.dump(k.value) for k in new.keywords}
allowed_changes = {'k_cache', 'v_cache', 'page_table'}
for name, value in oldkw.items():
    if name not in allowed_changes:
        assert newkw[name] == value, name
assert len(new.keywords) == len(old.keywords) + 1
assert [k.value.id for k in new.keywords if k.arg is None] == ['staged_kwargs', 'kwargs']
controls['native_kwargs_unchanged_except_KV_and_page_table'] = True
controls['preserved_native_arguments'] = sorted(k for k in oldkw if k and k not in allowed_changes)
controls['helper_sha256'] = hashlib.sha256((HERE / 'contiguous_prefill_helper.py').read_bytes()).hexdigest()
controls['passed'] = True
(HERE / 'contiguous_hook_cpu.json').write_text(json.dumps(controls, indent=2))
print(json.dumps(controls, indent=2))
