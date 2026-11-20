"""CPU layout proof. These tests do not execute Triton or prove kernel speed."""
import ast
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import forward_oracles as oracle
import cpu_bridge_model as bridge

here = Path(__file__).resolve().parent
root = here.parents[1]
started = time.time()
rows = []
# Cross-check the independent vector Humming oracle against the old scalar
# native forward-ownership loop, by extracting only that NumPy function.
old = root / 'research_2026-11-18_rl/expert_slab/test_slab.py'
node = next(x for x in ast.parse(old.read_text(encoding='utf-8-sig')).body if getattr(x, 'name', None) == 'pack')
class TorchStub:
    @staticmethod
    def from_numpy(x): return x
ns = {'np': np, 'torch': TorchStub}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(old), 'exec'), ns)
rng = np.random.default_rng(930173)
for wg in (False, True):
    q = rng.integers(0, 16, (2, 128, 192), dtype=np.uint8)
    sf = rng.integers(0, 256, (2, 128, 12), dtype=np.uint8)
    result, _ = oracle.humming_forward(q, sf, wg)
    assert np.array_equal(result, ns['pack'](q, wg).view(np.uint32))

storage = np.full(2 * 8192 * 2304 + 128, 0xA5, np.uint8)
for seed in (930181, 930197):
    rng = np.random.default_rng(seed)
    for n in (8192, 4096):
        e = 2
        q = rng.integers(0, 16, (e, n, 4096), dtype=np.uint8)
        sf = rng.integers(0, 256, (e, n, 256), dtype=np.uint8)
        q[0, 0, :16] = np.arange(16, dtype=np.uint8)
        sf[0, 0] = np.arange(256, dtype=np.uint8)
        packed = oracle.wgs_forward(q, sf, n == 8192)
        packed_hash = hashlib.sha256(packed).hexdigest()
        for layout in ('normal', 'humming_mma', 'humming_wgmma'):
            full = e * n * 2304
            before = storage[64 + full:].copy()
            a, b = bridge.convert(packed, storage[64:-64], e, n, layout)
            if layout == 'normal':
                ga, gb = oracle.normal_forward(q, sf)
            else:
                ga, gb = oracle.humming_forward(q, sf, layout == 'humming_wgmma')
            assert np.array_equal(a, ga.view(np.uint8).reshape(-1)), (seed, n, layout, 'codes')
            assert np.array_equal(b, gb.view(np.uint8).reshape(-1)), (seed, n, layout, 'scales')
            assert np.all(storage[:64] == 0xA5)
            assert np.array_equal(storage[64 + full:], before)
            assert hashlib.sha256(packed).hexdigest() == packed_hash
            rows.append(dict(seed=seed, stage='w13' if n == 8192 else 'w2', experts=e,
                layout=layout, nibble_values=q.size, scale_bytes=sf.size, exact=True,
                head_tail_canaries=True, shared_stage_overwrite=True, source_unchanged=True))
            print(json.dumps(rows[-1]), flush=True)
            del ga, gb, before
        # Independent source changes at remote expert/row/K boundaries, checking
        # exactly one destination byte per changed coefficient/scale.
        q[1, n - 1, 4095] ^= 8
        sf[0, 17, 255] ^= 128
        changed = oracle.wgs_forward(q, sf, n == 8192)
        assert np.count_nonzero(changed != packed) == 2
        ca, cb = bridge.convert(changed, storage[64:-64], e, n, 'normal')
        ga, gb = oracle.normal_forward(q, sf)
        assert np.array_equal(ca, ga.reshape(-1))
        assert np.array_equal(cb, gb.reshape(-1))
        del q, sf, packed, changed, ga, gb
frozen = root / 'research_2026-11-19/moe/normal_package/candidate.diff'
assert hashlib.sha256(frozen.read_bytes()).hexdigest() == '80db9ce3d958c58e31be4b64f06442d0ef152493221c3927ad9c43deab68319f'
report = dict(complete=True, scope='CPU arithmetic/layout proof only; actual Triton CUDA pending',
    rows=rows, source_changes=4, scalar_humming_oracle_crosscheck=True,
    all_16_code_values=True, all_256_scale_bytes=True,
    nibble_values=sum(x['nibble_values'] for x in rows), scale_bytes=sum(x['scale_bytes'] for x in rows),
    elapsed_seconds=time.time()-started,
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [here/'wgs_bridge.py', here/'forward_oracles.py', here/'cpu_bridge_model.py']})
(here/'bridge_cpu.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k != 'rows'}), flush=True)
