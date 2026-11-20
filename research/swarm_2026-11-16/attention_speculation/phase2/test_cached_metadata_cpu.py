"""Cache-key/control proof using exact exported wrapper, no CUDA imports."""
import ast, hashlib, json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace as NS
from test_persistent_cpu import Tensor, Kernel

P = Path(__file__).resolve().parent
REL = 'python/sglang/kernels/ops/attention/metadata.py'
OLD = P / 'persistent_metadata/candidate_sources' / REL
NEW = P / 'persistent_metadata_cached/candidate_sources' / REL

def run():
    def functions(path): return {n.name: n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef)}
    old, new = functions(OLD), functions(NEW)
    assert set(new) - set(old) == {'_decode_metadata_sm_count'}
    for name in old:
        if name != 'normal_decode_set_metadata': assert ast.dump(old[name]) == ast.dump(new[name]), name
    calls = []
    def properties(index):
        assert isinstance(index, int)
        calls.append(index)
        return NS(multi_processor_count={0: 16, 1: 132}[index])
    kernel = Kernel()
    scope = dict(torch=NS(cuda=NS(get_device_properties=properties)), lru_cache=lru_cache,
                 triton=NS(cdiv=lambda a, b: (a + b - 1) // b), _fused_decode_metadata_kernel=kernel)
    selected = [new['_decode_metadata_sm_count'], new['normal_decode_set_metadata']]
    unit = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(NEW), 'exec'), scope)
    fn = scope['normal_decode_set_metadata']
    batch, width = 32, 131072
    cache, cu = Tensor((batch,)), Tensor((batch + 1,))
    table, reqtable, req = Tensor((batch, width)), Tensor((batch, width)), Tensor((batch,))
    seq = Tensor((batch,))
    count = 0
    for index, repetitions in ((0, 1000), (1, 200), (0, 100)):
        seq.device = NS(type='cuda', index=index)
        for _ in range(repetitions):
            fn(cache, cu, table, reqtable, req, width, seq, 8, 1)
            assert kernel.grid == ((32, 2) if index == 0 else (32, 17))
            count += 1
    assert calls == [0, 1]
    seq.device = NS(type='xpu', index=None)
    fn(cache, cu, table, reqtable, req, width, seq, 8, 1)
    assert kernel.grid == (32, 512) and calls == [0, 1]
    seq.device = NS(type='cuda', index=0)
    fn(cache, cu, table, reqtable, req, width, seq, 8, 1, skip_page_table=True)
    assert kernel.grid == (1, 1) and calls == [0, 1]
    try: scope['_decode_metadata_sm_count'](None)
    except AssertionError: pass
    else: raise AssertionError('Ambiguous current-device key accepted')
    return dict(passed=True, repeated_wrapper_cases=count, queried_explicit_device_indices=calls,
                unchanged_kernel_ast=True, non_cuda_grid_preserved=True, skip_grid_preserved=True,
                ambiguous_device_rejected=True, candidate_sha256=hashlib.sha256(NEW.read_bytes()).hexdigest(),
                scope='Exact wrapper and hardware-count helper with mocked device properties; no GPU execution or timing')

if __name__ == '__main__':
    report = run()
    (P / 'persistent_metadata_cached/cpu_validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))
