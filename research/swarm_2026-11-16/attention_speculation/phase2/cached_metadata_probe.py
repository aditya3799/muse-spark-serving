"""Separate actual wrapper/glue wall time from repeated-body metadata GPU time."""
import argparse, hashlib, json, statistics, time
from pathlib import Path
from types import SimpleNamespace as NS
import metadata_probe as base

P = Path(__file__).resolve().parent
NEW = P / 'persistent_metadata_cached/candidate_sources/python/sglang/kernels/ops/attention/metadata.py'
PINS = dict(base.PINS)
del PINS[base.NEW]
PINS[NEW] = 'cd4e4f8bad3aa9cdcdcfd7e39b2e342f518cbe81c80ed01c865ccbc3d61e2653'
PINS[P / 'metadata_probe.py'] = '43b08e3b818681be17782cd34f1f191971a5daca235b8207dc98492f9495997c'

def kernel_only(torch, old, new, out):
    report = dict(status='running', passed=False, rows=[],
                  scope='SM86 repeated32-body metadata GPU graph timings; one host graph launch, original scalar prefix and full live page writes. Same buffers reused within graph; first-body cache pressure then warm replay, not whole serving cache residency or H200 speed.')
    def save(): (out / 'kernel_only.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    save()
    table = torch.arange(36 * 131072, device='cuda', dtype=torch.int32).view(36, 131072)
    flush = torch.empty(64 * 1024 * 1024, device='cuda', dtype=torch.uint8)
    repeats = 32
    for batch in (1, 4, 16, 32):
      for length in (2048, 74000):
        req = torch.arange(batch, device='cuda', dtype=torch.int32)
        lens = torch.full((batch,), length, device='cuda', dtype=torch.int32)
        fb = NS(batch_size=batch, req_pool_indices=req, seq_lens=lens, seq_lens_cpu=None,
                encoder_lens=None, forward_mode=base.Mode(), spec_info=NS(), kv_loc_plan=object(), out_cache_loc=None)
        graphs, owners, memories = {}, {}, {}
        for name, kernel in (('A', old), ('B', new)):
            b = base.backend(torch, kernel)
            b.req_to_token = table
            m = NS(cache_seqlens_int32=torch.zeros(batch, device='cuda', dtype=torch.int32),
                   cu_seqlens_k=torch.zeros(batch + 1, device='cuda', dtype=torch.int32),
                   page_table=torch.full((batch, 131072), -19, device='cuda', dtype=torch.int32),
                   swa_page_table=None, scheduler_metadata=None)
            b.target_verify_metadata[batch] = m
            for _ in range(3): b.init_forward_metadata_out_graph(fb)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(repeats): b.init_forward_metadata_out_graph(fb)
            graphs[name], owners[name], memories[name] = graph, b, m
        for step, new_length in enumerate((1, 131064, length)):
            lens.fill_(new_length)
            req.copy_((torch.arange(batch, device='cuda', dtype=torch.int32) + step + 1) % 36)
            for graph in graphs.values(): graph.replay()
            torch.cuda.synchronize()
            for field in ('cache_seqlens_int32', 'cu_seqlens_k', 'page_table'):
                assert torch.equal(getattr(memories['A'], field), getattr(memories['B'], field)), (batch, length, field)
        def sample(graph):
            flush.zero_()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); graph.replay(); end.record(); end.synchronize()
            return start.elapsed_time(end) * 1000 / repeats
        labels = ['A1', 'B1', 'A2', 'B2']
        samples = {name: [] for name in labels}
        for order in (labels, labels[::-1]):
            for label in order: sample(graphs[label[0]])
        for rep in range(12):
            for label in (labels if rep % 2 == 0 else labels[::-1]): samples[label].append(sample(graphs[label[0]]))
        med = {name: statistics.median(values) for name, values in samples.items()}
        arms = {name: statistics.median(samples[name + '1'] + samples[name + '2']) for name in 'AB'}
        controls = {name: med[name + '1'] / med[name + '2'] for name in 'AB'}
        row = dict(batch=batch, length=length, graph_body_repeats=repeats, samples_us=samples,
                   medians_us=arms, aa_ratios=controls, noise_pass=all(abs(v - 1) <= .10 for v in controls.values()),
                   speedup=arms['A'] / arms['B'], saved_kernel_us=arms['A'] - arms['B'])
        report['rows'].append(row); save()
        print(json.dumps({k: v for k, v in row.items() if k != 'samples_us'}), flush=True)
        del graphs, owners, memories, b, m, graph
        torch.cuda.synchronize()
    report.update(status='completed', passed=True, candidate_sha256=PINS[NEW])
    save()
    return report

def main():
    pa = argparse.ArgumentParser(); pa.add_argument('--out', type=Path, required=True)
    args, _ = pa.parse_known_args(); args.out.mkdir(parents=True, exist_ok=True)
    for path, digest in PINS.items(): assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    import torch
    old, new = base.load(base.OLD, 'cached_probe_old_metadata'), base.load(NEW, 'cached_probe_new_metadata')
    # Pure GPU first, then the identical prior four-arm wall harness. Both save
    # each shape, so a bounded exit never converts partial results into a pass.
    kernel_report = kernel_only(torch, old, new, args.out)
    base.NEW, base.PINS = NEW, PINS
    base.main()
    path = args.out / 'results.json'; report = json.loads(path.read_text())
    report['wrapper_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['kernel_only_report_sha256'] = hashlib.sha256((args.out / 'kernel_only.json').read_bytes()).hexdigest()
    report['variant_labels'] = {'AE': 'original eager', 'AG': 'original glue', 'PE': 'cached persistent eager', 'PG': 'cached persistent glue'}
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

if __name__ == '__main__': main()
