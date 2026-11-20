"""Exact exported metadata A/B. No FA4 attention or acceptance claim."""
import argparse, ast, hashlib, importlib.util, json, statistics, sys, time
from pathlib import Path
from types import SimpleNamespace as NS

P = Path(__file__).resolve().parent
OWN = P.parent
ROOT = OWN.parent.parent
FA = OWN / 'combined_attention_optional_v1/candidate/python/sglang/srt/layers/attention/flashattention_backend.py'
OLD = ROOT / 'sglang/python/sglang/kernels/ops/attention/metadata.py'
NEW = OWN / 'combined_attention_optional_v1/candidate/python/sglang/kernels/ops/attention/metadata.py'
GLUE = ROOT / 'sglang/python/sglang/srt/model_executor/runner/metadata_glue_graph.py'
PINS = {FA: '0b0d0d4288e7bab6c7c4fa3ec9c1068c75e52ddd019ff9eebf24cb3790b50b9e',
        OLD: '8aa5db57763f5ecfb01a05beba74791d0bf3921952c789d8a8432abf25c231ca',
        NEW: '300cb9428d6281c10812be0a302e88d82dff9f81baf5cab1087730fe8bc6df3c',
        GLUE: '30b64a0f0ff9ed882cbb7ba7b733b142a8d857b76fea2b27e3175751cfb57892'}

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

class Mode:
    def is_extend(self): return False
    def is_decode_or_idle(self): return False
    def is_target_verify(self): return True
    def is_draft_extend_v2(self): return False
    def is_dllm_extend(self): return False

def backend(torch, kernel):
    tree = ast.parse(FA.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FlashAttentionBackend')
    names = {'glue_draft_metadata', '_set_decode_page_metadata', '_apply_cuda_graph_metadata',
             '_init_full_cg_decode_metadata', 'init_forward_metadata_out_graph'}
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(torch=torch, SpeculativeAlgorithm=NS(from_string=lambda s: NS(is_dflash=lambda: s == 'DFLASH')),
                 get_spec=lambda: NS(speculative_algorithm='DFLASH'), get_parallel=lambda: NS(attn_cp_size=1),
                 _should_disable_scheduler_metadata_precompute=lambda: False,
                 normal_decode_set_metadata=kernel.normal_decode_set_metadata)
    unit = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(FA), 'exec'), scope)
    b = scope['FlashAttentionBackend']()
    vars(b).update(is_draft_runner=True, fa_impl_ver=4, topk=0, speculative_num_draft_tokens=8,
                  is_encoder_decoder=False, use_mla=False, local_attn_builder=None,
                  use_sliding_window_kv_pool=False, _kv_shard_pool=None, needs_cpu_seq_lens=False,
                  is_prefill_aware_swa=False, has_swa=True, sliding_window_size=2048,
                  kv_index_translator=NS(is_translating=False, reads_are_translated=False),
                  page_size=1, max_num_pages=131072, max_context_len=131072,
                  attn_backend_list=None, target_verify_metadata={})
    b._capture_unified_verify_metadata = lambda fb: False
    assert b.glue_draft_metadata() == ''
    return b

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=12)
    args = parser.parse_args()
    assert args.samples >= 6 and args.samples % 2 == 0
    args.out.mkdir(parents=True, exist_ok=True)
    for path, digest in PINS.items(): assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    import torch
    old, new = load(OLD, 'phase2_old_metadata'), load(NEW, 'phase2_new_metadata')
    glue_type = load(GLUE, 'phase2_actual_glue').MetadataGlueGraph
    report = dict(status='running', passed=False, device=torch.cuda.get_device_name(),
                  source_hashes={str(p): h for p, h in PINS.items()},
                  glue_sha256=hashlib.sha256(GLUE.read_bytes()).hexdigest(),
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=[],
                  scope='Exact target-verify metadata methods and real MetadataGlueGraph only; SM86 timings, no FA4 attention, model acceptance or H200 performance proof')
    def save(): (args.out / 'results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    save()
    start = time.monotonic()
    table = torch.arange(36 * 131072, dtype=torch.int32, device='cuda').view(36, 131072)
    for bs in (1, 4, 16, 32):
      for length in (2048, 74000):
        if time.monotonic() - start > 210: raise TimeoutError('Probe soft bound')
        req = torch.arange(bs, dtype=torch.int32, device='cuda')
        lens = torch.full((bs,), length, dtype=torch.int32, device='cuda')
        fb = NS(batch_size=bs, req_pool_indices=req, seq_lens=lens,
                seq_lens_cpu=torch.full((bs,), length, dtype=torch.int32), encoder_lens=None,
                forward_mode=Mode(), spec_info=NS(), kv_loc_plan=object(), out_cache_loc=None)
        arms, memories, glues = {}, {}, []
        for name, kernel, use_glue in [('AE', old, False), ('AG', old, True), ('PE', new, False), ('PG', new, True)]:
            b = backend(torch, kernel)
            b.req_to_token = table
            m = NS(cache_seqlens_int32=torch.zeros(bs, dtype=torch.int32, device='cuda'),
                   cu_seqlens_k=torch.zeros(bs + 1, dtype=torch.int32, device='cuda'),
                   page_table=torch.full((bs, 131072), -19, dtype=torch.int32, device='cuda'),
                   swa_page_table=None, scheduler_metadata=None)
            b.target_verify_metadata[bs] = m
            if use_glue:
                glue = glue_type(torch.device('cuda'))
                fn = lambda b=b, g=glue: g.run(b, fb, ('draft', bs))
                glues.append(glue)
            else:
                fn = lambda b=b: b.init_forward_metadata_out_graph(fb)
            for _ in range(4): fn()
            arms[name] = fn
            memories[name] = (b, m)
        torch.cuda.synchronize()
        checks = 0
        for step, value in enumerate((1, 2047, 2048, 2049, 74000, 131064, length)):
            req.copy_((torch.arange(bs, device='cuda', dtype=torch.int32) + step + 1) % 36)
            lens.fill_(value)
            # Keep page identifiers valid while changing the persistent pool contents.
            table.copy_(table.roll(1, dims=0))
            fb.kv_loc_plan = object()
            fb.seq_lens_cpu.fill_(value + 99)
            for fn in arms.values(): fn()
            torch.cuda.synchronize()
            baseline = memories['AE'][1]
            for name, (b, m) in memories.items():
                assert b.forward_metadata is m
                for attr in ('cache_seqlens_int32', 'cu_seqlens_k', 'page_table'):
                    assert torch.equal(getattr(baseline, attr), getattr(m, attr)), (name, attr, step)
                assert torch.equal(m.cache_seqlens_int32, lens + 8)
                assert torch.equal(m.cu_seqlens_k, torch.nn.functional.pad((lens + 8).cumsum(0, dtype=torch.int32), (1, 0)))
                # Every live page, not padded/undefined attention rows.
                assert torch.equal(m.page_table[:, :value + 8], table[req.long(), :value + 8])
                checks += 1
        assert all(g.is_captured(('draft', bs)) and not g.disabled for g in glues)
        # Same-input repeated calls measure only this metadata seam. Wall time includes
        # host dispatch and one synchronization per group; events include GPU idle gaps.
        labels = [n + suffix for suffix in ('1', '2') for n in arms]
        samples = {n: [] for n in labels}
        loops = 32
        def sample(fn):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(loops): fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1e6 / loops
        for n in labels: sample(arms[n[:-1]])
        for rep in range(args.samples):
            for n in (labels if rep % 2 == 0 else list(reversed(labels))):
                samples[n].append(sample(arms[n[:-1]]))
        med = {n: statistics.median(v) for n, v in samples.items()}
        controls = {n: med[n + '1'] / med[n + '2'] for n in arms}
        central = {n: statistics.median(samples[n + '1'] + samples[n + '2']) for n in arms}
        row = dict(batch=bs, length=length, metadata_comparisons=checks, samples_us=samples,
                   median_wall_us=central, aa_ratios=controls,
                   noise_pass=all(abs(x - 1) <= .10 for x in controls.values()),
                   glue_speedup=central['AE'] / central['AG'],
                   prefix_eager_speedup=central['AE'] / central['PE'],
                   prefix_glued_speedup=central['AG'] / central['PG'])
        report['rows'].append(row)
        save()
        print(json.dumps({k: v for k, v in row.items() if k != 'samples_us'}), flush=True)
        del arms, memories, glues, b, m, fn, glue
        torch.cuda.synchronize()
    report.update(status='completed', passed=True, elapsed_seconds=time.monotonic() - start)
    save()

if __name__ == '__main__': main()
