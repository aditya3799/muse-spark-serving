"""Actual helper byte gate; forced arch is explicit fixture injection only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'contiguous_kv_unapplied/sources/python/sglang'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


os.environ['SGLANG_EXPERIMENTAL_CONTIGUOUS_PREFILL_KV'] = '1'
helper = load('contiguous_diagnostic', ROOT / 'srt/layers/attention/contiguous_prefill_kv.py')
pack_module = load('sglang.kernels.ops.kvcache.contiguous_kv_staging', ROOT / 'kernels/ops/kvcache/contiguous_kv_staging.py')
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


def stamp():
    return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())


def verify_all(staged, key, value, table, lengths):
    for b, length in enumerate(lengths.cpu().tolist()):
        for start in range(0, table.shape[1], 4096):
            stop = min(start + 4096, table.shape[1])
            valid_stop = max(start, min(stop, length))
            if valid_stop > start:
                ids = table[b, start:valid_stop].long()
                for got, original in zip(staged, (key, value)):
                    expected = original.view(torch.uint8)[:, 0][ids]
                    assert torch.equal(got[b, start:valid_stop].view(torch.uint8), expected)
            if stop > valid_stop:
                for got in staged:
                    assert bool((got[b, valid_stop:stop].view(torch.uint8) == 0).all())


def main():
    torch.manual_seed(26011989)
    report = {'actual_capability': list(torch.cuda.get_device_capability()),
              'fixture_injection': 'Capability is mocked to9 ONLY to exercise byte-staging helper; no native FP8 attention runs.',
              'scope': 'Persistent-budget scaffold with dummy905977856B arena; excludes native FFN intermediates and simultaneous attention native scratch.',
              'helper_sha256': hashlib.sha256((ROOT / 'srt/layers/attention/contiguous_prefill_kv.py').read_bytes()).hexdigest(),
              'cases': []}
    backend = types.SimpleNamespace(fa_impl_ver=3, page_size=1, kv_cache_is_mxfp8=False,
                                    use_mla=False, fa_skip_kv_cache=False)
    layer = types.SimpleNamespace(is_cross_attention=False, head_dim=64, v_head_dim=64,
                                  tp_k_head_num=2, tp_v_head_num=2, tp_q_head_num=16)
    fb = types.SimpleNamespace(forward_mode='extend', dcp_kv_mask=None)
    physical = 8192
    key = torch.arange(physical * 128, device='cuda', dtype=torch.int32).to(torch.uint8).reshape(physical, 1, 2, 64).view(torch.float8_e4m3fn)
    value = (key.view(torch.uint8) ^ 173).view(torch.float8_e4m3fn)
    q = torch.zeros((753, 1024), device='cuda', dtype=torch.float8_e4m3fn)
    table = torch.zeros((1, 127), device='cuda', dtype=torch.int32)
    lengths = torch.tensor([127], device='cuda', dtype=torch.int32)
    original_args = (backend, q, key, value, table, lengths, layer, fb)
    before = stamp()
    assert helper.maybe_stage(*original_args, allowed=True) is None
    assert not hasattr(backend, '_contiguous_prefill_kv_state')
    report['real_sm86_guard_no_allocation'] = stamp()['allocated'] == before['allocated']
    dummy = torch.empty(905977856, device='cuda', dtype=torch.uint8)
    report['dummy_moe_arena_bytes'] = dummy.numel()
    report['before_kv_arena'] = stamp()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), patch.object(helper, 'is_hopper', return_value=True):
        for index, (batch, cap) in enumerate(((1,127), (3,8193), (4,131072), (7,65537), (1,257))):
            table = torch.randint(0, physical, (batch, cap), device='cuda', dtype=torch.int32)
            lens = [max(0, cap - b * (cap // max(batch,1))) for b in range(batch)]
            if batch > 1:
                lens[-1] = 0
            lengths = torch.tensor(lens, device='cuda', dtype=torch.int32)
            for b, length in enumerate(lens):
                table[b, length:] = -1234567
            args = (backend, q, key, value, table, lengths, layer, fb)
            started = time.perf_counter()
            staged = helper.maybe_stage(*args, allowed=True)
            torch.cuda.synchronize()
            first_use_ms = (time.perf_counter() - started) * 1000
            assert staged is not None
            pointer = backend._contiguous_prefill_kv_state[1].data_ptr()
            verify_all(staged, key, value, table, lengths)
            # Eager-break protocol: capture a producer and a consumer separately;
            # metadata is changed on host, staging runs eager between replays.
            pre = torch.cuda.CUDAGraph()
            with torch.cuda.graph(pre, stream=stream):
                key.view(torch.uint8).bitwise_xor_(37)
            checksum = torch.empty((), device='cuda', dtype=torch.int64)
            post = torch.cuda.CUDAGraph()
            with torch.cuda.graph(post, stream=stream):
                checksum.copy_(staged[0].view(torch.uint8)[..., 0, 0].sum())
            for _ in range(2):
                lengths.sub_(1).clamp_(min=0)
                table[:, 0] = 3
                pre.replay()
                staged_new = helper.maybe_stage(*args, allowed=True)
                post.replay()
                verify_all(staged_new, key, value, table, lengths)
                assert checksum.item() == staged_new[0].view(torch.uint8)[..., 0, 0].sum().item()
                assert backend._contiguous_prefill_kv_state[1].data_ptr() == pointer
            flags.dp.capturing_prefill_graph = True
            assert helper.maybe_stage(*args, allowed=True) is None
            flags.dp.capturing_prefill_graph = False
            report['cases'].append(dict(batch=batch, capacity=cap, first_call_ms=first_use_ms,
                                        all_bytes_exact=True, eager_break_replay_exact=True,
                                        storage_pointer=pointer, memory=stamp(),
                                        compiled_variants=len(pack_module._pack.device_caches[torch.cuda.current_device()][0])))
        other_stream = torch.cuda.Stream()
        with torch.cuda.stream(other_stream):
            assert helper.maybe_stage(*args, allowed=True) is None
        report['wrong_stream_native_fallback'] = True
        table_bad = torch.empty((4,131073), device='cuda', dtype=torch.int32)
        lengths_bad = torch.zeros((4,), device='cuda', dtype=torch.int32)
        assert helper.maybe_stage(backend,q,key,value,table_bad,lengths_bad,layer,fb,allowed=True) is None
        report['one_row_over_capacity_native_fallback'] = True
    torch.cuda.synchronize()
    report['after'] = stamp()
    report['owned_persistent_extra_bytes'] = dummy.numel() + helper.MAX_BYTES
    report['compiled_variants'] = len(pack_module._pack.device_caches[torch.cuda.current_device()][0])
    assert report['compiled_variants'] == 1, 'context-dependent JIT specialization remains'
    report['passed'] = True
    (HERE / 'contiguous_helper_gpu.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}, indent=2))


if __name__ == '__main__':
    main()
