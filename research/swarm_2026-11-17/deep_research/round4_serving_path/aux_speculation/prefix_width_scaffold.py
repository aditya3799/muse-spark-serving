"""Isolated target-prefix preparation scaffold; no serving integration.
Uses source-extracted DFlashVerifyInput, KVLocPlan methods, eager FA3 metadata,
and graph width guard. Worker/ForwardBatch init/kernel launch are explicit fakes.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass
from enum import Enum, auto
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS, ModuleType
from typing import Optional
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SRT = ROOT / 'sglang/python/sglang/srt'
SOURCES = {
 'info': SRT / 'speculative/dflash_info.py',
 'plan': SRT / 'mem_cache/kv_loc_plan.py',
 'fa3': SRT / 'layers/attention/flashattention_backend.py',
 'graph': SRT / 'model_executor/runner/decode_cuda_graph_runner.py',
 'result': SRT / 'managers/scheduler_components/batch_result_processor.py',
}

def node(key, name, parent=None):
    tree = ast.parse(SOURCES[key].read_text())
    body = tree.body if parent is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == parent).body
    return next(n for n in body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)

def load(nodes, namespace):
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[]))
    exec(compile(module, '<source-extracted>', 'exec'), namespace)

class ForwardMode(Enum):
    IDLE = auto()
    TARGET_VERIFY = auto()
    DECODE = auto()
    def is_idle(self): return self is self.IDLE
    def is_decode_or_idle(self): return self in (self.IDLE, self.DECODE)
    def is_target_verify(self): return self is self.TARGET_VERIFY
    def is_draft_extend_v2(self): return False
    def is_extend(self): return False

class IdSpaceKind(Enum):
    FULL = auto()
    SLIDING_WINDOW = auto()

class SpecInput:
    def __init__(self, spec_input_type): self.spec_input_type = spec_input_type

class ForwardBatch:
    @classmethod
    def init_new(cls, batch, runner, **kwargs):
        # Explicit fake: real prepare_for_verify controls the kwargs and plan bind.
        fb = NS(**vars(batch))
        fb.capture_hidden_mode = kwargs['capture_hidden_mode']
        fb.positions = fb.spec_info.positions
        fb.batch_size = len(fb.seq_lens)
        fb.encoder_lens = None
        fb.replace_embeds = None
        plan = kwargs['kv_loc_plan']
        plan.bind(fb, runner.translator, cols=kwargs['write_cols'])
        return fb

namespace = dict(globals())
namespace.update({'SpecInputType': NS(DFLASH_VERIFY='DFLASH_VERIFY'),
                  'CaptureHiddenMode': NS(FULL='FULL'), '_is_npu': False})
load([node('info', 'DFlashVerifyInput')], namespace)
DFlashVerifyInput = namespace['DFlashVerifyInput']
load([node('fa3', 'FlashAttentionMetadata')], namespace)
Metadata = namespace['FlashAttentionMetadata']
namespace['FlashAttentionMetadata'] = Metadata
load([node('fa3', 'init_forward_metadata', 'FlashAttentionBackend'),
      node('fa3', '_apply_cuda_graph_metadata', 'FlashAttentionBackend')], namespace)
load([node('result', '_get_speculative_output_stride')], namespace)
output_stride = namespace['_get_speculative_output_stride']

class TokenSpan:
    def __init__(self, start, stop): self.start, self.stop = start, stop

plan_ns = dict(globals())
load([node('plan', n, 'KVLocPlan') for n in ('bind', '_cols', 'write_ids', 'virtual_write_ids')], plan_ns)
class SourcePlan:
    bind = plan_ns['bind']
    _cols = plan_ns['_cols']
    write_ids = plan_ns['write_ids']
    virtual_write_ids = plan_ns['virtual_write_ids']
    def __init__(self, reqs, virtual, reader):
        self.req_pool_indices, self.write_virtual = reqs, virtual
        self._write_ids = {reader.space(kind).key: reader.transform(virtual, kind) for kind in IdSpaceKind}
        self.read_extent = 8
    def is_translated_for(self, reader): return reader.is_translating

class Reader:
    def __init__(self, table, translated):
        self.table = table
        self.is_translating = translated
    def space(self, kind): return NS(key=kind)
    def transform(self, values, kind):
        return values * (2 if kind is IdSpaceKind.FULL else 3) + 1 if self.is_translating else values
    def read_table(self, plan, rows, kind=IdSpaceKind.FULL):
        return NS(ids=self.transform(self.table[plan.req_pool_indices[:rows]], kind).to(torch.int32))

class Backend:
    init_forward_metadata = namespace['init_forward_metadata']
    _apply_cuda_graph_metadata = namespace['_apply_cuda_graph_metadata']
    def __init__(self, width, table, reader, swa=False):
        self.speculative_num_draft_tokens = width
        self.topk = 1
        self.max_context_len = table.shape[1]
        self.req_to_token = table
        self.req_to_token_pool = NS(req_to_token=table)
        self.kv_index_translator = reader
        self.use_sliding_window_kv_pool = swa
        self._kv_shard_pool = None
        self.page_size = 1
        self.target_verify_metadata = {}
        self.last_delta = None
    def _maybe_init_local_attn_metadata(self, metadata, device): pass
    def _get_swa_write_locations(self, fb):
        return fb.kv_loc_plan.write_ids(self.kv_index_translator, kind=IdSpaceKind.SLIDING_WINDOW, cols=fb.kv_loc_cols)
    def _set_decode_page_metadata(self, plan, metadata, reqs, lens, delta):
        # Explicit fake for device kernel: record source-selected delta, model
        # resulting values on CPU. Never claim normal_decode kernel executed.
        self.last_delta = delta
        metadata.cache_seqlens_int32.copy_((lens + delta).to(torch.int32))
        metadata.page_table = self.kv_index_translator.read_table(plan, len(reqs)).ids

# Exact source guard, followed by True solely to represent other guard success.
guard_src = node('graph', 'can_run_graph', 'DecodeCudaGraphRunner')
guard = next(n for n in guard_src.body if isinstance(n, ast.If) and 'captured_req_width' in ast.unparse(n.test))
guard_fn = ast.FunctionDef(name='width_guard', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='self'), ast.arg(arg='forward_batch')], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[ast.Assign(targets=[ast.Name(id='spec_info', ctx=ast.Store())], value=ast.Attribute(value=ast.Name(id='forward_batch', ctx=ast.Load()), attr='spec_info', ctx=ast.Load())), guard, ast.Return(value=ast.Constant(value=True))], decorator_list=[])
load([guard_fn], namespace)
class Graph:
    can_run_graph = namespace['width_guard']
    def __init__(self, width, backend): self.captured_req_width, self.backend, self.loads = width, backend, 0
    def load_batch(self, fb):
        self.loads += 1
        n, width = fb.batch_size, self.captured_req_width
        self.backend.target_verify_metadata[n] = Metadata(cache_seqlens_int32=torch.zeros(n, dtype=torch.int32), cu_seqlens_q=torch.arange(0, (n+1)*width, width, dtype=torch.int32))
        self.backend._apply_cuda_graph_metadata(fb.kv_loc_plan, n, fb.req_pool_indices, fb.seq_lens, None, fb.forward_mode, fb.spec_info, fb.seq_lens_cpu, fb.out_cache_loc)

@dataclass(frozen=True)
class PrefixWidth:
    draft: int = 8
    verify: int = 8
    def __post_init__(self):
        if self.draft < 1 or self.verify < 1 or self.verify > self.draft:
            raise ValueError('verify must be in [1,draft]')
    def stage(self, ids, positions, plan):
        if ids.ndim != 2 or ids.shape[1] != self.draft or positions.shape != ids.shape:
            raise ValueError('draft IDs and positions must retain full draft width')
        if self.verify == self.draft:
            ids_flat, pos_flat, cols = ids.view(-1), positions.view(-1), None
        else:
            ids_flat = ids[:, :self.verify].contiguous().view(-1)
            pos_flat = positions[:, :self.verify].contiguous().view(-1)
            cols = slice(0, self.verify)
        return DFlashVerifyInput(draft_token=ids_flat, positions=pos_flat, draft_token_num=self.verify, kv_loc_plan=plan, kv_loc_cols=cols)

def main():
    fake_mod = ModuleType('sglang.srt.speculative.spec_utils')
    fake_mod.prepare_mamba_track_for_verify = lambda batch: None
    sys.modules[fake_mod.__name__] = fake_mod
    report = {'status': 'CPU_EXTRACTED_PREPARATION_ONLY', 'cases': 0, 'default8_identity_cases': 0,
              'negative_controls': {}, 'sources': {k: hashlib.sha256(v.read_bytes()).hexdigest() for k,v in SOURCES.items()},
              'real_source_executed': ['DFlashVerifyInput dataclass and prepare_for_verify', 'KVLocPlan bind/write_ids/_cols/virtual_write_ids', 'FA3 full eager init_forward_metadata', 'FA3 full _apply_cuda_graph_metadata', 'DecodeCudaGraphRunner exact width guard fragment', 'scheduler _get_speculative_output_stride'],
              'fakes': ['ForwardBatch.init_new field packaging', 'worker and model runners', 'Mamba hook no-op; no Mamba admission', 'graph load plumbing; no CUDA graph', 'normal_decode metadata kernel modeled on CPU', 'virtual-to-physical translation arithmetic fixture'],
              'limits': ['No full model forward', 'No CUDA graph/runtime', 'No performance or lossless serving qualification', 'No adaptive policy']}
    for bs, width, translated, graph_path in itertools.product((1, 2, 3, 8, 17, 32), (1, 2, 4, 6, 8), (False, True), (False, True)):
        table = torch.arange(40 * 192, dtype=torch.int64).view(40, 192) + 100
        reqs = torch.arange(bs, dtype=torch.int64).flip(0)
        prefix = torch.arange(bs, dtype=torch.int64) + 20
        ids = torch.arange(bs*8, dtype=torch.int64).view(bs, 8) + 900
        positions = prefix[:, None] + torch.arange(8)
        virtual = table[reqs[:, None], positions].flatten()
        reader = Reader(table, translated)
        plan = SourcePlan(reqs, virtual, reader)
        config = PrefixWidth(8, width)
        info = config.stage(ids, positions, plan)
        backend = Backend(width, table, reader, swa=translated)
        graph = Graph(width, backend) if graph_path else None
        worker = NS(model_runner=NS(attn_backend=backend, decode_cuda_graph_runner=graph, translator=reader))
        batch = NS(forward_mode=ForwardMode.DECODE, seq_lens=prefix, seq_lens_cpu=prefix.clone(), req_pool_indices=reqs, out_cache_loc=virtual, encoder_lens=None)
        fb, graph_ok = info.prepare_for_verify(batch, worker)
        assert graph_ok == graph_path
        assert fb.input_ids.tolist() == ids[:, :width].flatten().tolist()
        assert fb.positions.tolist() == positions[:, :width].flatten().tolist()
        expected_slots = reader.transform(virtual.view(bs, 8)[:, :width].flatten(), IdSpaceKind.FULL)
        assert torch.equal(fb.out_cache_loc, expected_slots)
        assert torch.equal(backend.forward_metadata.cache_seqlens_int32, (prefix+width).to(torch.int32))
        assert backend.forward_metadata.cu_seqlens_q.tolist() == list(range(0, (bs+1)*width, width))
        assert info.num_tokens_per_req == width and info.num_tokens_for_logprob_per_req == width
        assert output_stride(NS(speculative_output_stride=width, speculative_num_draft_tokens=8)) == width
        assert plan.read_extent == 8 and ids.shape == (bs, 8)
        assert virtual.numel() == bs*8  # allocation reservation is unchanged
        if width == 8:
            assert info.draft_token.data_ptr() == ids.data_ptr()
            assert info.positions.data_ptr() == positions.data_ptr()
            assert fb.out_cache_loc.data_ptr() == plan._write_ids[IdSpaceKind.FULL].data_ptr()
            assert info.kv_loc_cols is None
            report['default8_identity_cases'] += 1
        if not graph_path and translated:
            expected_swa = reader.transform(virtual.view(bs,8)[:, :width].flatten(), IdSpaceKind.SLIDING_WINDOW)
            assert torch.equal(backend.forward_metadata.swa_out_cache_loc, expected_swa)
        report['cases'] += 1
    # Negative: width metadata alone cannot repair an existing width8 backend.
    backend.speculative_num_draft_tokens = 8
    info4 = PrefixWidth(8,4).stage(ids, positions, plan)
    fb4 = NS(**vars(fb)); fb4.spec_info=info4; fb4.input_ids=info4.draft_token; fb4.positions=info4.positions
    backend.init_forward_metadata(fb4)
    assert backend.forward_metadata.cu_seqlens_q[-1].item() == bs*8 != bs*4
    report['negative_controls']['unchanged_fa3_backend_uses_width8'] = {'input_rows': bs*4, 'metadata_q_rows': bs*8}
    assert not Graph(8, backend).can_run_graph(fb4)
    report['negative_controls']['existing_graph_rejects_width4'] = True
    try:
        PrefixWidth(8,9)
        raise AssertionError('invalid width admitted')
    except ValueError: pass
    report['negative_controls']['width_outside_draft_rejected'] = True
    (HERE/'PREFIX_PREPARATION_RESULT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    import itertools
    main()
