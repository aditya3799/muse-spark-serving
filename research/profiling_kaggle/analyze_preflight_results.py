"""Read notebook-01 artifacts; write a separate, reproducible interpretation."""
import collections
import hashlib
import json
import statistics
import sys
from pathlib import Path

p = Path(sys.argv[1]).resolve()
checks = json.loads((p / 'checksums.json').read_text())
integrity = {}
for name, digest in checks.items():
    path = (p / name).resolve()
    assert path.parent == p, name
    integrity[name] = hashlib.sha256(path.read_bytes()).hexdigest() == digest
assert all(integrity.values()), integrity
report = json.loads((p / 'report.json').read_text())
assert report['source_manifest'] == json.loads((p / 'source_manifest.json').read_text())
events = json.loads((p / 'routing_trace.json').read_text())['traceEvents']
annotations = [e for e in events if e.get('cat') == 'user_annotation']
runtime = {e['args']['correlation']: e for e in events
           if e.get('cat') == 'cuda_runtime' and 'correlation' in e.get('args', {})}
groups = collections.defaultdict(list)
unassigned = []
for event in events:
    if event.get('cat') not in ('kernel', 'gpu_memcpy'):
        continue
    launch = runtime.get(event.get('args', {}).get('correlation'))
    matches = [i for i, a in enumerate(annotations) if launch
               and (a['pid'], a['tid']) == (launch['pid'], launch['tid'])
               and a['ts'] <= launch['ts'] < a['ts'] + a['dur']]
    if len(matches) != 1:
        unassigned.append(event['name'])
    else:
        groups[matches[0]].append(event)
assert not unassigned, unassigned
samples = collections.defaultdict(list)
for i, a in enumerate(annotations):
    g = groups[i]
    samples[a['name']].append({
        'cpu_range_us': a['dur'],
        'kernels': sum(e['cat'] == 'kernel' for e in g),
        'copies': sum(e['cat'] == 'gpu_memcpy' for e in g),
        'kernel_sum_us': sum(e['dur'] for e in g if e['cat'] == 'kernel'),
        'copy_sum_us': sum(e['dur'] for e in g if e['cat'] == 'gpu_memcpy'),
    })
assert len(samples) == 3 and all(len(v) == 6 for v in samples.values())
summary = {name: {key: statistics.median(s[key] for s in rows[1:])
                  for key in rows[0]} for name, rows in samples.items()}
analysis = {'run_id': report['run_id'], 'integrity': integrity,
            'source_manifest_matches': True, 'trace_categories': dict(collections.Counter(e.get('cat', 'uncategorized') for e in events)),
            'attribution': 'Device correlation ID -> CPU launch -> enclosing diagnostic range on same process/thread. Kernel/copy durations only; exclude GPU annotation ranges.',
            'aggregation': 'Keep all six raw samples; summarize iterations 2-6 because the first trace iteration has a large CPU instrumentation/startup outlier. These are profiler-on measurements, not benchmark latency.',
            'trace_samples': samples, 'trace_medians_iterations_2_to_6': summary}
(p / 'PREFLIGHT_ANALYSIS.json').write_text(json.dumps(analysis, indent=2) + '\n', encoding='utf-8')
lines = ['KAGGLE NOTEBOOK 01: REVIEW', 'Run: ' + report['run_id'], '',
    'VALIDATION', 'All eight downloaded artifact checksums passed. Source manifest matches local V4 notebook.',
    'Two Tesla T4 GPUs detected; this notebook used GPU 0 only.',
    'FP32/FP16 checks, changing-input CUDA graph replay, and 30 routing correctness cases passed.',
    'Original V4 Triton alignment and native BF16 were skipped as intended on this T4.', '',
    'WARM DIAGNOSTIC TIMINGS (microseconds, CUDA event median; p10-p90)',
    'These intervals include exposed gaps between GPU operations. They are not pure kernel time.']
for row in report['sections']['routing_diagnostic']['timing_cases']:
    lines.append(f"{row['tokens']:3d} tokens, {row['pattern']:9s}: {row['event_us_median']:.1f} us ({row['event_us_p10']:.1f}-{row['event_us_p90']:.1f}); synchronized wall median {row['wall_us_median']:.1f} us")
lines += ['', 'TRACE BREAKDOWN: 128 TOKENS, HOT-LOCAL ROUTING',
    'Median per stage over trace iterations 2-6, in microseconds.',
    'Stage                         Kernels Copies Kernel time Copy time']
for name, s in summary.items():
    lines.append(f"{name.split('/', 1)[-1]:29s} {s['kernels']:7g} {s['copies']:6g} {s['kernel_sum_us']:11.2f} {s['copy_sum_us']:9.2f}")
lines += ['', 'What each stage does:',
    '- Map: convert global expert IDs to IDs for this GPU; preserve -1 for inactive routes.',
    '- Sort/count: group routes by expert and count how many belong to each expert.',
    '- Padding counts: round counts to multiples of eight; this diagnostic does not build the packed buffer.', '',
    'INTERPRETATION',
    '1. The profiling setup works, including GPU kernel tracing.',
    '2. Increasing input from 8 to 128 tokens barely changes this diagnostic elapsed time. That supports investigating fixed dispatch/allocation/instrumentation overhead; it does not establish its exact cost.',
    '3. Each trace iteration contains 17 kernels and 3 device copies. The profiler reports 138 device events across six iterations: 102 kernels, 18 copies, and 18 annotation ranges. Those 138 events are NOT 138 kernels.',
    '4. Do not sum profile_operations.csv into percentages: it includes GPU annotation ranges and multiple representations of the same work. All 120 actual kernel/copy events were attributed via launch correlation.',
    '5. First trace map CPU range was 11.637 ms; later map CPU ranges were about 0.30-0.37 ms. This is a profiler-on outlier, not measured serving startup latency.',
    '6. The unprofiled benchmark and instrumented trace are separate runs. Do not subtract traced kernel time from benchmark time to claim an exact CPU overhead percentage.',
    '7. Fixed-block synthetic padding can be large: at 8 tokens, uniform routing has 3 live routes and 24 active padded rows; hot-local has 37 and 72. This alone says nothing about actual GEMM time or useful gains from a different block size.', '',
    'NEXT EXPERIMENT',
    'Notebook 02 should compare the same supported routing diagnostic with annotations disabled: warm eager versus CUDA graph replay, with matching allocation semantics and changing-input correctness checks. Profile in a separate pass. Isolate the original supported PyTorch expert-ID mapping to distinguish project code from diagnostic sorting/counting overhead.',
    'Keep current V4 serving code unchanged until a candidate has correctness evidence and a relevant performance comparison.', '',
    'SCOPE LIMITS',
    'This is adapted synthetic PyTorch routing, not measured stock/V4 alignment. It excludes packed-buffer construction, expert GEMMs, attention, inter-GPU communication, scheduler startup, tokenizer startup, and full-model serving.',
    'Therefore these results neither confirm scheduler/tokenizer initialization as the main bottleneck nor predict H200 timings, serving percentages, or contest speedup.',
    'The presence of nvcc/nsys/ncu does not prove usable compilation targets or hardware-counter permissions.', '']
(p / 'PREFLIGHT_REVIEW.txt').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps({'checksums_verified': len(integrity), 'attributed_device_operations': sum(map(len, groups.values())), 'stage_medians': summary, 'written': ['PREFLIGHT_ANALYSIS.json', 'PREFLIGHT_REVIEW.txt']}, indent=2))
