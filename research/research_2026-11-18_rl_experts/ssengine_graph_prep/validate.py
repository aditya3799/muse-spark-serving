"""Exercise the frozen and patched worker dispatch with CPU buffers and instrumentation."""
from __future__ import annotations
import ast

import copy
import enum
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import numpy as np

here = Path(__file__).resolve().parent
repo = here.parents[1] / "sglang"
manifest = json.loads((here / "manifest.json").read_text())
prefix = "python/sglang/srt/"


def source(tree, path):
    return subprocess.check_output(["git", "-C", str(repo), "show", tree + ":" + prefix + path]).decode()


def method(tree, path, name):
    nodes = [n for n in ast.walk(ast.parse(source(tree, path))) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, (path, name)
    return nodes[0]


def materialize(node, scope):
    n = copy.deepcopy(node)
    n.decorator_list = []
    code = ast.unparse(n)
    exec("from __future__ import annotations\n" + code, scope)
    return scope[n.name]


class Mode(enum.Enum):
    IDLE = 0
    TARGET_VERIFY = 1

    def is_idle(self):
        return self == Mode.IDLE


class Buffer(np.ndarray):
    def copy_(self, other):
        np.copyto(self, other)


class Forward(SimpleNamespace):
    @classmethod
    def init_new(cls, batch, runner, **kw):
        return cls(input_ids=batch.input_ids, positions=batch.spec_info.positions,
                   batch_size=len(batch.seq_lens_cpu), lens=batch.seq_lens_cpu, seq_sum=batch.seq_lens_sum,
                   forward_mode=batch.forward_mode, can_run_decode_cuda_graph=batch.allowed,
                   original_global_num_tokens_cpu=batch.global_counts, slot=batch.out_cache_loc,
                   forward_metadata_ready=False, mamba=batch.mamba, opts=tuple(kw.values()))


def mock_import(name, **symbols):
    mod = ModuleType(name)
    mod.__dict__.update(symbols)
    sys.modules[name] = mod


def set_mamba(batch):
    batch.mamba = True


mock_import("sglang.srt.speculative.spec_utils", prepare_mamba_track_for_verify=set_mamba)
mock_import("sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks",
            maybe_build_dsv4_verify_bundle=lambda *args, **kw: None)


def exercise(tree, scenario):
    device, dp, has_graph, allowed, mixed, zero_rank = scenario
    scope = dict(_is_npu=(device == "npu"), is_cuda=lambda: device == "cuda",
                 get_parallel=lambda: SimpleNamespace(attn_dp_enabled=dp),
                 ForwardBatch=Forward, ForwardMode=Mode, _skip_attn_backend_init_warned=True)
    meta = "model_executor/forward_batch_info.py"
    for name in ["mark_forward_metadata_ready", "needs_forward_metadata_init",
                 "apply_deprecated_skip_attn_backend_init"]:
        setattr(Forward, name, materialize(method(tree, meta, name), scope))
    graph_file = "model_executor/runner/decode_cuda_graph_runner.py"
    copy_inputs = materialize(method(tree, graph_file, "_copy_replay_inputs"), scope)
    graph_source = source(tree, graph_file)
    for anchor in ("self.buffer_registry.fill_from(", "if not forward_batch.needs_forward_metadata_init():",
                   "self._copy_replay_inputs(forward_batch)", "self.model_runner.spec_algorithm.is_dspark()",
                   "self.load_batch(forward_batch, pp_proxy_tensors)"):
        assert anchor in graph_source
    eager = source(tree, "model_executor/runner/eager_runner.py")
    assert "or forward_batch.forward_mode.is_target_verify()" in eager
    info = materialize(method(tree, "speculative/dflash_info.py", "prepare_for_verify"), scope)
    worker = method(tree, "speculative/dflash_worker_v2.py", "forward_batch_generation")
    start = next(i for i,n in enumerate(worker.body) if ast.unparse(n).startswith("seq_lens_cpu_backup ="))
    end = next(i for i,n in enumerate(worker.body) if ast.unparse(n).startswith("target_out = self.target_worker.forward_batch_generation"))
    func = ast.parse("def worker_call(self, batch, verify_input, draft_input, block_size): pass").body[0]
    func.body = worker.body[start:end+1] + [ast.Return(value=ast.Name(id="target_out",ctx=ast.Load()))]
    invoke = materialize(func, scope)
    counts = dict(graph_load=0, static_full_fill=0, preplanned_recopy=0, recopy_bytes=0, eager_init=0)

    class Graph:
        def __init__(self):
            self.buffers = SimpleNamespace(input_ids=np.full(256, -9, dtype='int64').view(Buffer),
                                           positions=np.full(256, -8, dtype='int64').view(Buffer))
            self._prefilled_inputs = None

        def can_run_graph(self, fb):
            return fb.can_run_decode_cuda_graph

        def load_batch(self, fb):
            counts["graph_load"] += 1
            if not fb.needs_forward_metadata_init():
                counts["preplanned_recopy"] += 1
                counts["recopy_bytes"] += fb.input_ids.nbytes + fb.positions.nbytes
                copy_inputs(self, fb)
                return
            counts["static_full_fill"] += 1
            self.raw_num_token = fb.input_ids.size
            self.buffers.input_ids[:self.raw_num_token] = fb.input_ids
            self.buffers.positions[:self.raw_num_token] = fb.positions
            self.metadata = (fb.lens.copy(), fb.seq_sum, fb.slot.copy(), fb.mamba, fb.opts)

    def eager_init(fb):
        counts["eager_init"] += 1
        graph.metadata = fb.lens.copy(), fb.seq_sum, fb.slot.copy(), fb.mamba, fb.opts

    graph = Graph()
    class Target:
        def forward_batch_generation(self, *, forward_batch, skip_attn_backend_init, **kw):
            fb = forward_batch
            fb.apply_deprecated_skip_attn_backend_init(skip_attn_backend_init)
            if has_graph and graph.can_run_graph(fb):
                graph.load_batch(fb)
                ids = graph.buffers.input_ids[:fb.input_ids.size]
                pos = graph.buffers.positions[:fb.input_ids.size]
            else:
                eager_init(fb)  # the real eager target-verify branch always replans
                ids, pos = fb.input_ids, fb.positions
            return (ids.tolist(), pos.tolist(), graph.metadata[0].tolist(), graph.metadata[1],
                    graph.metadata[2].tolist(), graph.metadata[3], graph.metadata[4])
    target = Target()
    target.model_runner = SimpleNamespace(decode_cuda_graph_runner=graph if has_graph else None,
                                         attn_backend=SimpleNamespace(init_forward_metadata=eager_init))
    outer = SimpleNamespace(target_worker=target)
    class Spec:
        prepare_for_verify = info
        capture_hidden_mode = "full"
        kv_loc_plan = 33
        kv_loc_cols = 8
        draft_token_num = 8
        live_seq_lens_cpu = None
    spec = Spec()
    rng = np.random.default_rng(119249)
    ids_storage = np.empty(256, dtype='int64')
    pos_storage = np.empty_like(ids_storage)
    outputs = []
    for step, size in enumerate([1,4,16,32,7,31,7,4,1,32]):
        n = 8*size
        ids_storage[:n] = rng.integers(0, 120000, n)
        pos_storage[:n] = rng.integers(1700, 86000, n)
        spec.draft_token = ids_storage[:n]
        spec.positions = pos_storage[:n]
        lens = rng.integers(500, 80000, size)
        slots = rng.integers(0, 2_000_000, n)
        batch = SimpleNamespace(seq_lens_cpu=lens, seq_lens_sum=int(lens.sum()), mamba=False,
                                is_extend_in_batch=mixed, forward_mode=Mode.TARGET_VERIFY, allowed=allowed,
                                global_counts=[0, n] if zero_rank else [n,n],out_cache_loc=slots)
        outputs.append(invoke(outer,batch,spec,SimpleNamespace(nxt_kv_lens_cpu=None),8))
        assert batch.seq_lens_cpu is lens and batch.seq_lens_sum == int(lens.sum())
        assert outputs[-1][:5] == (spec.draft_token.tolist(), spec.positions.tolist(), (lens+8).tolist(),
                                   int((lens+8).sum()), slots.tolist())
    return outputs, counts


scenarios = {
    "cuda_target_graph": ("cuda",False,True,True,False,False),
    "cuda_eager_ineligible": ("cuda",False,True,False,False,False),
    "cuda_no_runner": ("cuda",False,False,False,False,False),
    "cuda_dp_graph": ("cuda",True,True,True,False,False),
    "cuda_dp_idle_peer": ("cuda",True,True,True,False,True),
    "cuda_dp_mixed_extend": ("cuda",True,True,True,True,False),
    "npu_deferred": ("npu",False,True,True,False,False),
    "other_device_eager": ("other",False,False,False,False,False),
    "other_device_graph": ("other",False,True,True,False,False)}
report = {}
for name, setup in scenarios.items():
    a, old = exercise(manifest["parent_tree"], setup)
    b, new = exercise(manifest["target_tree"], setup)
    assert a == b
    if name == "cuda_target_graph":
        assert old["preplanned_recopy"] == 10 and new["preplanned_recopy"] == 0
        assert old["static_full_fill"] == new["static_full_fill"] == 10
    else:
        assert old == new, name
    report[name] = dict(outputs_match=True, changed_input_steps=len(b), parent=old, candidate=new)
(here / "cpu_validation.json").write_text(json.dumps(report, indent=2))
print("Passed", len(report), "scenarios;", report["cuda_target_graph"])
