"""Exact runner/glue Python control flow with fake CUDA recording, no GPU claims."""
import ast, contextlib, hashlib, json, re, textwrap
from pathlib import Path
from types import SimpleNamespace as NS

P = Path(__file__).resolve().parent
SGL = P.parents[2] / 'sglang/python/sglang'
RUNNER = SGL / 'srt/model_executor/runner/decode_cuda_graph_runner.py'
GLUE = SGL / 'srt/model_executor/runner/metadata_glue_graph.py'

class Array:
    def __init__(self, values, root=None, count=None):
        self.root = self if root is None else root
        if root is None: self.values = list(values)
        self.count = len(values) if count is None else count
    def __getitem__(self, key):
        assert isinstance(key, slice) and key.start is None and key.step is None
        return Array([], self.root, key.stop)
    @property
    def shape(self): return (self.count,)
    def read(self): return self.root.values[:self.count]

class FakeCUDA:
    def __init__(self): self.recording = None; self.fail_next = False
    def Stream(self): return object()
    def CUDAGraph(self):
        class Graph:
            def __init__(self): self.ops = []; self.replays = 0
            def replay(self):
                for op in self.ops: op()
                self.replays += 1
        return Graph()
    @contextlib.contextmanager
    def graph(self, graph, **kwargs):
        self.recording = graph
        try:
            yield
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError('injected fake capture failure')
        finally: self.recording = None

def exact_class(path, name, selected=None, scope=None):
    if selected is not None:
        # Native Python 3.8 cannot parse unrelated modern parenthesized-with
        # methods. Extract selected method text unchanged, then parse it.
        cls = ast.ClassDef(name=name, bases=[], keywords=[],
                           body=[extract_function(path, n, 4) for n in sorted(selected)], decorator_list=[])
    else:
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        cls.bases = []
    unit = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path), 'exec'), scope)
    return scope[name]

def extract_function(path, name, indent):
    lines = path.read_text(encoding='utf-8-sig').splitlines(keepends=True)
    begin = next(i for i, line in enumerate(lines) if line.startswith(' ' * indent + 'def ' + name + '('))
    end = len(lines)
    # Header can span lines ending at the original def indentation. Find the
    # next def/decorator/class at equal or lower indentation, not header lines.
    for i in range(begin + 1, len(lines)):
        line = lines[i]
        if line.strip() and len(line) - len(line.lstrip()) <= indent and re.match(r'\s*(?:def |class |@)', line):
            end = i; break
    return ast.parse(textwrap.dedent(''.join(lines[begin:end]))).body[0]

def run():
    cuda = FakeCUDA()
    scope = dict(torch=NS(cuda=cuda), logger=NS(warning=lambda *a, **kw: None),
                 SimpleNamespace=NS, DpPaddingMode=NS(get_default_mode_in_cuda_graph=lambda: 'fixed'))
    Glue = exact_class(GLUE, 'MetadataGlueGraph', scope=scope)
    view_fn = extract_function(RUNNER, 'build_replay_fb_view', 0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), view_fn], type_ignores=[])), str(RUNNER), 'exec'), scope)
    Runner = exact_class(RUNNER, 'DecodeCudaGraphRunner',
                         {'_init_replay_attn_metadata', '_init_replay_attn_metadata_from_view', '_glue_draft_inputs'}, scope)
    class Backend:
        def __init__(self, pool):
            self.attn_backend_list = None
            self.pool = pool
            self.metadata = {}
            self.calls = 0
        def init_forward_metadata_out_graph(self, fb):
            self.calls += 1
            m = self.metadata.setdefault(fb.batch_size, NS(value=None))
            self.forward_metadata = m
            req, lens, loc, pool = fb.req_pool_indices, fb.seq_lens, fb.out_cache_loc, self.pool
            def op():
                m.value = (req.read(), lens.read(), loc.read(), list(pool))
            if cuda.recording is None: op()
            else: cuda.recording.ops.append(op)
    def make_runner(pool):
        r = Runner()
        r.buffers = NS(**{name: Array(list(range(32))) for name in
                         ('input_ids', 'positions', 'req_pool_indices', 'seq_lens', 'seq_lens_cpu', 'out_cache_loc', 'encoder_lens')},
                       mamba_track_indices=None)
        r.buffers.out_cache_loc = Array(list(range(256)))
        r.buffers.input_ids = Array(list(range(256)))
        r.buffers.positions = Array(list(range(256)))
        r._metadata_glue = Glue('fake-cuda')
        r._draft_metadata_glue = True
        r._target_metadata_glue = False
        r._skip_glued_replay_view = False
        r.enable_two_batch_overlap = r.enable_pdmux = False
        r.model_runner = NS(lora_manager=None)
        r.capture_forward_mode = 'verify'
        r.seq_len_fill_value = 1
        r.is_encoder_decoder = False
        r._global_num_tokens_for_graph = lambda n: None
        r._prep_attention_variant = lambda variant: contextlib.nullcontext()
        b = Backend(pool)
        r._replay_attn_backend = lambda: b
        return r, b
    def batch(step, bs):
        return NS(forward_mode='verify', seq_lens_sum=step, seq_lens_cpu=None if step % 2 else Array([99] * 32),
                  max_seq_len_host=99, out_cache_loc=Array([-999] * (bs * 8)), out_cache_loc_virtual=None,
                  kv_loc_plan=object(), kv_loc_cols=object(), max_seq_len_override=None, spec_info=object(),
                  lora_ids=None, can_run_decode_cuda_graph=True)
    def invoke(r, b, bs, step, raw_bs=None):
        for name in ('req_pool_indices', 'seq_lens', 'out_cache_loc'):
            buf = getattr(r.buffers, name)
            buf.values[:] = [step * 100 + i for i in range(buf.count)]
        b.pool[:] = [step, step + 1]
        fb = batch(step, bs)
        r._init_replay_attn_metadata(fb, raw_bs=bs if raw_bs is None else raw_bs,
                                    bs=bs, num_tokens=bs * 8, glue_key=('verify', bs),
                                    early_staged=False, attention_variant=None)
        expected_loc = (r.buffers.out_cache_loc.read()[:bs * 8] if
                        not r._metadata_glue.disabled and raw_bs in (None, bs)
                        and not r.enable_two_batch_overlap and not r.enable_pdmux
                        and r.model_runner.lora_manager is None else fb.out_cache_loc.read())
        # Eager fallback deliberately keeps the fresh batch's location tensor;
        # this is safe outside capture and is the original runner contract.
        assert b.forward_metadata.value == (r.buffers.req_pool_indices.read()[:bs],
                                           r.buffers.seq_lens.read()[:bs], expected_loc, list(b.pool))
    r, b = make_runner([0, 1])
    count = 0
    for step, bs in enumerate((1, 4, 1, 4, 1, 4, 8, 8, 8, 1, 4, 8), 1):
        invoke(r, b, bs, step); count += 1
    assert r._metadata_glue.replays > 0
    for step in range(13, 17):
        r._skip_glued_replay_view = True
        invoke(r, b, 4, step); count += 1
    for attr in ('enable_two_batch_overlap', 'enable_pdmux'):
        setattr(r, attr, True)
        before = r._metadata_glue.replays
        invoke(r, b, 4, count + 1); count += 1
        assert r._metadata_glue.replays == before
        setattr(r, attr, False)
    r.model_runner.lora_manager = NS(attn_dp_enabled=False)
    invoke(r, b, 4, 25); count += 1
    r.model_runner.lora_manager = None
    invoke(r, b, 4, 26, raw_bs=3); count += 1
    # New key warmups, failed capture, then a previously successful key. The
    # caller's disabled guard must prevent reusing either captured state.
    r._skip_glued_replay_view = False
    invoke(r, b, 2, 30); invoke(r, b, 2, 31)
    cuda.fail_next = True
    # Capture fallback uses the already repointed stable location view this call.
    fb = batch(32, 2)
    calls_before_failure = b.calls
    r._init_replay_attn_metadata(fb, raw_bs=2, bs=2, num_tokens=16,
                                glue_key=('verify', 2), early_staged=False, attention_variant=None)
    assert r._metadata_glue.disabled
    assert b.calls == calls_before_failure + 2
    assert b.forward_metadata.value == (r.buffers.req_pool_indices.read()[:2],
                                       r.buffers.seq_lens.read()[:2], r.buffers.out_cache_loc.read()[:16], list(b.pool))
    before = r._metadata_glue.replays
    invoke(r, b, 1, 33)
    assert r._metadata_glue.replays == before
    count += 4
    # Supported recapture builds a fresh runner, so replacement pool addresses
    # belong to new graph captures; arbitrary live replacement is not promised.
    r2, b2 = make_runner([100, 101])
    for step in (34, 35, 36, 37): invoke(r2, b2, 1, step); count += 1
    assert r2._metadata_glue.is_captured(('verify', 1))
    return dict(passed=True, route_cases=count, source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (RUNNER, GLUE)},
                scope='Exact Python runner/view/glue control with simulated device recording and injected capture failure; no CUDA execution or allocator/arithmetic proof')

if __name__ == '__main__':
    result = run()
    (P / 'glue_routing_cpu.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))
