"""Verify and summarize a notebook-02 ZIP without extracting its members."""
import csv
import hashlib
import io
import json
import statistics
import sys
import zipfile
from pathlib import Path

archive, out = map(Path, sys.argv[1:3])
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist()) == len(set(z.namelist())), 'Duplicate archive members'
    hashes = json.loads(z.read('checksums.json'))
    verified = {n: hashlib.sha256(z.read(n)).hexdigest() == h for n, h in hashes.items()}
    assert all(verified.values()), verified
    report = json.loads(z.read('report.json'))
    def table(name):
        return list(csv.DictReader(io.StringIO(z.read(name).decode())))
    timings, raw, cases = map(table, ('timings.csv', 'raw_samples.csv', 'benchmark_cases.csv'))
    stages, device = map(table, ('trace_stages.csv', 'trace_device_operations.csv'))
root = archive.parent
local_manifest = json.loads((root / 'notebook02_support/source_manifest_02.json').read_text())
assert report['source_manifest'] == local_manifest
assert len(cases) == 50 and all(c['status'] == 'passed' for c in cases)
assert sum(int(c['changed_input_replays']) for c in cases) == 250
assert len(timings) == 110 and len(raw) == 1650
for row in timings:
    sample = [r for r in raw if all(r[k] == row[k] for k in ('tokens', 'pattern', 'stage', 'variant'))]
    assert len(sample) == 15 and len({r['sample'] for r in sample}) == 15
    assert abs(statistics.median(float(r['event_us']) for r in sample) - float(row['event_us_median'])) < 1e-8
assert len(device) == 120
assert sum(r['category'] == 'kernel' for r in device) == 102
assert sum(r['category'] == 'gpu_memcpy' for r in device) == 18
assert all(r['stage'] != 'unattributed' for r in device)
def stats(values):
    v = sorted(values)
    return {'median': statistics.median(v), 'p10': v[int(.1*(len(v)-1))], 'p90': v[int(.9*(len(v)-1))]}
comparisons = []
for tokens in (1, 8, 9, 128, 256):
    for pattern in ('uniform', 'hot_local'):
        rows = [r for r in timings if (r['tokens'], r['pattern'], r['stage']) == (str(tokens), pattern, 'full_diagnostic')]
        med = {r['variant']: float(r['event_us_median']) for r in rows}
        paired = {}
        for r in raw:
            if (r['tokens'], r['pattern'], r['stage']) == (str(tokens), pattern, 'full_diagnostic'):
                paired.setdefault(r['sample'], {})[r['variant']] = float(r['event_us'])
        comparisons.append({'tokens': tokens, 'pattern': pattern, 'event_medians_us': med,
                            'ratio_of_eager_to_graph_medians': med['eager']/med['graph_replay'],
                            'paired_label_delta_us': stats([p['eager_annotated_control']-p['eager'] for p in paired.values()]),
                            'paired_eager_graph_ratio': stats([p['eager']/p['graph_replay'] for p in paired.values()])})
trace_summary = {}
for stage in sorted({r['stage'] for r in stages}):
    warm = [r for r in stages if r['stage'] == stage and int(r['iteration']) > 1]
    trace_summary[stage] = {key: statistics.median(float(r[key]) for r in warm)
                            for key in ('kernel_count', 'kernel_sum_us', 'copy_memset_count', 'copy_memset_sum_us')}
previous = json.loads((root / 'report.json').read_text())
analysis = {'run_id': report['run_id'], 'archive': str(archive.resolve()),
            'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'verified_checksums': verified,
            'manifest_matches': True, 'timing_medians_recomputed': 110, 'correctness_cases': 60,
            'benchmark_cases_passed': 50, 'changed_input_graph_checks': 250,
            'full_diagnostic_comparisons': comparisons, 'trace_medians_iterations_2_to_6': trace_summary,
            'previous_environment': previous['sections']['environment'], 'current_environment': report['sections']['environment'],
            'interpretation': 'Within-run diagnostic comparisons only. Graph replay changes host dispatch and allocation together. Paired sample ranges are descriptive, not confidence intervals. No H200 or serving speedup inference.'}
lines = ['NOTEBOOK 02: VERIFIED KAGGLE REVIEW', 'Run: ' + report['run_id'], '',
         'VALIDATION', 'All 10 member checksums and source manifest matched. All 110 event medians were recomputed from the 1,650 raw samples.',
         'All 60 eager route cases, 50 benchmark cases and 250 changed-input graph checks passed.',
         'GPU trace: 102 actual kernels and 18 device copies, all assigned to stages. Annotation spans excluded.', '',
         'FULL DIAGNOSTIC: median CUDA-event intervals, microseconds',
         'Tokens Pattern     Labeled eager  Plain eager  Graph replay  Eager/graph']
for r in comparisons:
    m = r['event_medians_us']
    lines.append(f"{r['tokens']:6d} {r['pattern']:11s} {m['eager_annotated_control']:12.2f} {m['eager']:12.2f} {m['graph_replay']:12.2f} {r['ratio_of_eager_to_graph_medians']:12.2f}")
lines += ['', 'FINDINGS',
    '1. Labels are measurable overhead. At 8 tokens/hot-local, labeled eager is 388.49 us versus 273.10 us plain eager. The median paired label delta is 116.35 us (descriptive p10-p90 102.63-130.15 us). This is harness overhead, not a serving-code optimization.',
    '2. Replay reduces the same 8-token diagnostic to 69.63 us, a 3.92x ratio of medians versus plain eager. At 128 tokens/hot-local, 276.25 us becomes 105.23 us (2.63x). Host dispatch/allocation overhead matters, but this does not identify a pure launch-cost percentage.',
    '3. Larger workloads become visible under replay: hot-local graph medians are 60.95 us at 1 token, 69.63 us at 8, and 105.23 us at 128. Eager medians over those cases remain around 268-276 us.',
    '4. Exact mapping helper at 8/hot-local: eager 76.35 us, replay 19.56 us. At 128/hot-local: eager 78.01 us, replay 24.68 us. These exclude the separately timed int64-to-int32 conversion.',
    '5. Synthetic sorting/counting is the largest kernel-duration sum in the separate 128-token trace. This is NOT the sorting implementation used by V4 alignment, so it is not evidence to optimize serving argsort.', '',
    'TRACE MEDIANS (iterations 2-6; profiler-on, microseconds)']
for name, values in trace_summary.items():
    lines.append(f"{name}: kernels={values['kernel_count']:g}, kernel sum={values['kernel_sum_us']:.2f} us, copies={values['copy_memset_count']:g}, copy sum={values['copy_memset_sum_us']:.2f} us")
lines += ['', 'NOISE AND COMPARABILITY',
    '- Notebook 01: torch 2.11.0+cu130, Triton 3.6.0, NumPy 2.1.3, driver 580.178.04. Notebook 02: torch 2.14.1+cu130, Triton 3.8.0, NumPy 2.4.6, driver 580.82.07. GPU count and clock snapshots also differ. Do not call cross-notebook timing differences a code speedup.',
    '- The 9-token hot-local mapping eager samples move from roughly 130-168 us early to roughly 71-92 us late while replay stays near 21 us. This points to host/runtime variability; it does not demonstrate a production 8/9-token threshold penalty.',
    '- 8-token uniform eager has p10-p90 about 264-439 us. The 256-token hot-local case also has elevated times. Repeat candidate comparisons with alternating order and more than one trial before acceptance.',
    '- Trace collection is separate and instrumented. Its GPU sums cannot be subtracted from the benchmark to compute CPU overhead. Isolated stage medians are not additive.', '',
    'RELEVANCE TO CURRENT V4 SOURCE',
    'sglang/python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py:409 gates an existing fused conversion/mapping/alignment path on an expert map, CUDA, contiguous inputs, and 1..64 routes.',
    'With eight routes per token and those conditions satisfied, 1..8 tokens already take the fused path. The standalone mapping helper benchmark therefore does not describe that serving path.',
    'Above 64 routes, the source retains a separate conversion and mapping helper before real alignment. That fallback is a more relevant place to test a mapping-only candidate.', '',
    'NEXT EXPERIMENT',
    'Notebook 03 should compare the exact fallback mapping helper with one equivalent PyTorch lookup formulation that avoids redundant intermediate work, focusing on 9/32/128/256 token cases. Verify negative sentinels, arbitrary local mappings, int32/int64 input handling, and changed-input graph replay. Measure both eager and replay in repeated, interleaved trials. Include any lookup-table preparation cost and its reuse assumptions.',
    'Do not widen the V4 fused-kernel threshold based on this diagnostic; its actual alignment and expert GEMM effects were not measured. Do not introduce a serving change until the isolated candidate is validated.',
    'Scheduler/tokenizer startup remains a separate unmeasured track. This notebook neither confirms nor rejects that startup hypothesis.', '']
out.mkdir(parents=True, exist_ok=True)
(out / 'NOTEBOOK02_ANALYSIS.json').write_text(json.dumps(analysis, indent=2), encoding='utf-8')
(out / 'NOTEBOOK02_REVIEW.txt').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps({'verified_members': len(verified), 'recomputed_medians': 110, 'trace_summary': trace_summary,
                  'output': str(out.resolve())}, indent=2))
