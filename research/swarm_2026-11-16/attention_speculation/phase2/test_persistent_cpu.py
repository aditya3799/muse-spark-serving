"""Exact wrapper grid and traversal ownership proof; no Triton compilation."""
import ast, hashlib, json, math, random
from pathlib import Path
from types import SimpleNamespace as NS

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
REL = 'python/sglang/kernels/ops/attention/metadata.py'
OLD = ROOT / 'sglang' / REL
NEW = P / 'persistent_metadata/candidate_sources' / REL

def nodes(path):
    return {n.name: n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef)}

class Tensor:
    def __init__(self, shape, device='cuda'):
        self.shape = shape; self.device = NS(type=device)
    def stride(self, dim=None):
        strides = (self.shape[1], 1) if len(self.shape) == 2 else (1,)
        return strides if dim is None else strides[dim]

class Kernel:
    def __getitem__(self, grid):
        def call(*args, **kwargs): self.grid, self.args, self.kwargs = grid, args, kwargs
        return call

def run():
    old, new = nodes(OLD), nodes(NEW)
    assert set(old) == set(new)
    for name in old:
        if name not in ('_fused_decode_metadata_kernel', 'normal_decode_set_metadata'):
            assert ast.dump(old[name]) == ast.dump(new[name]), name
    ko, kn = old['_fused_decode_metadata_kernel'], new['_fused_decode_metadata_kernel']
    owner_old = next(n for n in ko.body if isinstance(n, ast.If))
    owner_new = next(n for n in kn.body if isinstance(n, ast.If))
    assert ast.dump(owner_old) == ast.dump(owner_new)
    loop = next(n for n in kn.body if isinstance(n, ast.For))
    assignments = {n.targets[0].id: n.value for n in kn.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    for name in ('first_col', 'column_step'):
        assert 'int64' in ast.dump(assignments[name])
    assert 'int64' in ast.dump(loop.iter.args[1])
    body_begin = next(i for i, n in enumerate(ko.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'cols')
    assert [ast.dump(n) for n in ko.body[body_begin:]] == [ast.dump(n) for n in loop.body]
    sm = NS(value=16)
    kernel = Kernel()
    scope = dict(torch=NS(cuda=NS(get_device_properties=lambda device: NS(multi_processor_count=sm.value))),
                 triton=NS(cdiv=lambda a, b: (a + b - 1) // b), _fused_decode_metadata_kernel=kernel)
    unit = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), new['normal_decode_set_metadata']], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(NEW), 'exec'), scope)
    fn = scope['normal_decode_set_metadata']
    grids = 0
    for batch in (0, 1, 3, 8, 32, 129, 1025):
      for width in (0, 1, 127, 128, 129, 2048, 131072):
       for page in (1, 16, 256):
        for device in ('cuda', 'xpu'):
         for sms in (16, 132):
          for skip in (False, True):
           sm.value = sms
           tensors = [Tensor((batch,)), Tensor((batch + 1,)), Tensor((batch, width)),
                      Tensor((max(1, batch), max(1, width * page))), Tensor((batch,)), Tensor((batch,), device)]
           fn(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], width, tensors[5], 8, page, skip_page_table=skip)
           block = 256 if skip or page == 1 else 128
           columns = (width + block - 1) // block
           if device == 'cuda' and batch > 0 and not skip: columns = min(columns, max(1, (4 * sms + batch - 1) // batch))
           expected = (1, 1) if skip or width == 0 else (batch, columns)
           assert kernel.grid == expected, (batch, width, page, device, sms, skip)
           assert kernel.kwargs['BLOCK_COLS'] == block
           grids += 1
    rng = random.Random(216)
    traversal = 0
    geometries = [(w, b, c, live, tail) for w in (0, 1, 127, 128, 129, 2049, 131072)
                  for b in (128, 256) for c in (1, 3, 16, 64)
                  for live in (0, 1, w // 2, w) for tail in (False, True)]
    geometries += [(rng.randrange(5000), rng.choice((128, 256)), rng.randrange(1, 65), rng.randrange(5000), bool(rng.randrange(2))) for _ in range(500)]
    for width, block, columns, live, tail in geometries:
        live = min(width, live)
        end = width if tail else live
        writes = []
        for pid in range(columns):
            for start in range(pid * block, end, columns * block):
                writes.extend(range(start, min(start + block, end)))
        assert len(writes) == end and len(set(writes)) == end
        assert sorted(writes) == list(range(end))
        # Store tail values are zero: the unchanged resolver masks both page
        # and token mapping loads by live_mask, even when ZERO_TAIL stores them.
        for col in (0, max(0, live - 1), live, max(0, width - 1)):
            if col < end:
                owner = (col // block) % columns
                iteration = (col // block) // columns
                assert owner * block + iteration * columns * block <= col < (owner + 1) * block + iteration * columns * block
        traversal += 1
    wide = 0
    for bound in (2 ** 31 - 1, 2 ** 31, 2 ** 32 - 1):
      for block in (128, 256):
       for columns in (1, 3, 64, 528, (bound + block - 1) // block):
        for col in (0, bound // 2, bound - 1):
            owner = (col // block) % columns
            iteration = (col // block) // columns
            start = owner * block + iteration * columns * block
            assert start <= col < start + block
            assert start + columns * block < 2 ** 63
            wide += 1
    return dict(passed=True, wrapper_grid_cases=grids, traversal_cases=traversal, wide_integer_index_cases=wide,
                unchanged_prefix_owner=True, unchanged_mapping_and_store_body=True,
                source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (OLD, NEW)},
                limitations=['No Triton compile or GPU execution yet.', 'Inputs and output buffers must be distinct, as in audited serving callers; arbitrary aliasing not promised.', 'Four CTAs per SM is an unmeasured policy, not an established optimum.'])

if __name__ == '__main__':
    report = run()
    (P / 'persistent_metadata/cpu_validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))
