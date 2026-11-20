"""Offline routing overhead experiment. No model loading or serving changes."""
import csv
import hashlib
import html
import importlib.metadata
import json
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

MODES = ('uniform', 'hot_local', 'all_remote', 'all_local', 'sentinel')
STAGES = ('convert_i64_to_i32', 'original_mapping_i32', 'sort_and_count', 'padding_counts', 'full_diagnostic')


def make_routes(tokens, mode, seed=604):
    rng = random.Random(seed)
    remote = [e for e in range(256) if not 64 <= e < 96]
    ids = []
    for _ in range(tokens):
        if mode == 'all_local':
            row = rng.sample(range(64, 96), 8)
        elif mode == 'all_remote':
            row = rng.sample(remote, 8)
        elif mode == 'hot_local':
            hot = [64, 65, 66, 67]
            row = hot + rng.sample([e for e in range(256) if e not in hot], 4)
        else:
            row = rng.sample(range(256), 8)
            if mode == 'sentinel':
                row[0] = -1
        ids.extend(row)
    return ids


def cpu_reference(values, mapping):
    assert all(x == -1 or 0 <= x < len(mapping) for x in values)
    local = [-1 if x < 0 else mapping[x] for x in values]
    counts = [sum(x == e for x in local) for e in range(-1, 32)]
    ordered = sorted(range(len(values)), key=lambda i: local[i])
    return {'convert_i64_to_i32': (values,), 'original_mapping_i32': (local,),
            'sort_and_count': (ordered, counts), 'padding_counts': ([((x + 7) // 8) * 8 for x in counts],),
            'full_diagnostic': (local, ordered, counts, [((x + 7) // 8) * 8 for x in counts])}


def write_csv(path, rows):
    if rows:
        with Path(path).open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)


def statistics_us(samples):
    vals = sorted(samples)
    return {'median': statistics.median(vals), 'p10': vals[int(.1 * (len(vals) - 1))],
            'p90': vals[int(.9 * (len(vals) - 1))], 'min': min(vals), 'max': max(vals)}


class RoutingStudy:
    def __init__(self, manifest, helper_source, output_root=None, force_cpu=False,
                 samples=15, repeats=20, tokens=(1, 8, 9, 128, 256)):
        self.manifest, self.helper_source = manifest, helper_source
        assert hashlib.sha256(helper_source.encode()).hexdigest() == manifest['mapping_helper_sha256']
        self.samples, self.repeats, self.tokens = samples, repeats, tuple(tokens)
        assert samples >= 5 and repeats >= 1 and all(m >= 1 for m in tokens)
        self.force_cpu, self.torch, self.gpu = force_cpu, None, False
        root = Path(output_root) if output_root else (Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd())
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:6]
        self.out = root / ('v4_routing_overhead_' + run_id)
        self.out.mkdir(parents=True, exist_ok=False)
        self.report = {'schema_version': 2, 'run_id': run_id, 'source_manifest': manifest,
                       'config': {'samples': samples, 'repeats': repeats, 'tokens': list(tokens), 'patterns_timed': ['uniform', 'hot_local'], 'block_size': 8, 'top_k': 8, 'global_experts': 256, 'local_experts': 32},
                       'scope': 'Synthetic PyTorch routing plus exact project mapping helper; NOT V4 alignment, a serving benchmark, or an H200 prediction.',
                       'method': 'Warm repeated calls; no profiler in primary timings. Same operator sequence for eager and graph. Eager allocates through caching allocator; graph reuses private-pool storage. Input copies, graph capture and correctness checks excluded. Event intervals can contain launch gaps. Stage medians are not additive.',
                       'sections': {}}
        self.raw_samples, self.timings, self.case_results = [], [], []
        (self.out / 'mapping_helper.py').write_text(helper_source, encoding='utf-8')
        self.save()

    def save(self):
        (self.out / 'report.json').write_text(json.dumps(self.report, indent=2), encoding='utf-8')
        write_csv(self.out / 'timings.csv', self.timings)
        write_csv(self.out / 'raw_samples.csv', self.raw_samples)
        write_csv(self.out / 'benchmark_cases.csv', self.case_results)

    def section(self, name, fn):
        begin = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:
            result = {'status': 'failed', 'error': repr(exc), 'traceback': traceback.format_exc(limit=10)}
        result['wall_seconds'] = round(time.perf_counter() - begin, 3)
        self.report['sections'][name] = result
        self.save()
        print(name + ': ' + result['status'])
        print(json.dumps(result, indent=2))
        return result

    def environment(self):
        def run():
            import torch
            self.torch = torch
            self.gpu = torch.cuda.is_available() and not self.force_cpu
            namespace = {'torch': torch}
            exec(self.helper_source, namespace)
            self.mapping_helper = namespace['_map_topk_ids_to_local_experts']
            versions = {}
            for name in ('torch', 'triton', 'numpy', 'ipykernel'):
                try:
                    versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    versions[name] = None
            result = {'status': 'passed' if self.gpu else 'cpu_only', 'python': sys.version,
                      'os': platform.system(), 'packages': versions, 'torch_cuda': torch.version.cuda,
                      'gpu_devices': [], 'forced_cpu': self.force_cpu}
            if self.gpu:
                torch.cuda.set_device(0)
                for i in range(torch.cuda.device_count()):
                    p = torch.cuda.get_device_properties(i)
                    result['gpu_devices'].append({'index': i, 'name': p.name, 'capability': [p.major, p.minor], 'memory_bytes': p.total_memory})
                result['active_device'] = 0
                try:
                    smi = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,pstate,power.draw,clocks.sm', '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
                    result['nvidia_smi'] = smi.stdout.strip()
                except Exception as exc:
                    result['nvidia_smi_note'] = repr(exc)
            else:
                result['note'] = 'GPU timings will be skipped. Enable GPU T4 in Kaggle, restart, and Run All.'
            return result
        return self.section('environment', run)

    def inputs(self, tokens):
        t = self.torch
        device = 'cuda' if self.gpu else 'cpu'
        return {'raw': t.empty(tokens * 8, dtype=t.int64, device=device),
                'ids': t.empty(tokens * 8, dtype=t.int32, device=device),
                'mapping': t.empty(256, dtype=t.int32, device=device),
                'local': t.empty(tokens * 8, dtype=t.int32, device=device),
                'counts': t.empty(33, dtype=t.int64, device=device)}

    def load_inputs(self, inputs, tokens, mode, seed=604, reverse_mapping=False):
        t = self.torch
        values = make_routes(tokens, mode, seed)
        mapping = [-1] * 256
        mapping[64:96] = list(reversed(range(32))) if reverse_mapping else list(range(32))
        reference = cpu_reference(values, mapping)
        data = {'raw': values, 'ids': values, 'mapping': mapping,
                'local': reference['original_mapping_i32'][0], 'counts': reference['sort_and_count'][1]}
        for name, value in data.items():
            inputs[name].copy_(t.tensor(value, dtype=inputs[name].dtype))
        return reference

    def functions(self, inputs):
        t = self.torch
        raw, ids, mapping, local, counts = (inputs[k] for k in ('raw', 'ids', 'mapping', 'local', 'counts'))
        def sort_count(x):
            order = t.argsort(x, stable=True)
            hist = t.zeros(33, device=x.device, dtype=t.int64)
            hist.scatter_add_(0, x.long() + 1, t.ones_like(x, dtype=t.int64))
            return order, hist
        def padding(x):
            return ((x + 7) // 8) * 8
        def full():
            mapped = self.mapping_helper(raw.to(t.int32), mapping)
            order, hist = sort_count(mapped)
            return mapped, order, hist, padding(hist)
        def annotated():
            with t.profiler.record_function('diagnostic/map_global_to_local'):
                mapped = self.mapping_helper(raw.to(t.int32), mapping)
            with t.profiler.record_function('diagnostic/sort_and_count'):
                order, hist = sort_count(mapped)
            with t.profiler.record_function('diagnostic/padding_counts'):
                padded = padding(hist)
            return mapped, order, hist, padded
        return {'convert_i64_to_i32': lambda: raw.to(t.int32),
                'original_mapping_i32': lambda: self.mapping_helper(ids, mapping),
                'sort_and_count': lambda: sort_count(local), 'padding_counts': lambda: padding(counts),
                'full_diagnostic': full, 'annotated_control': annotated}

    def check(self, outputs, reference):
        values = outputs if isinstance(outputs, tuple) else (outputs,)
        assert len(values) == len(reference), 'Output tuple length mismatch'
        for i, (got, expected) in enumerate(zip(values, reference)):
            assert got.tolist() == expected, 'Independent CPU oracle mismatch in output ' + str(i)

    def correctness(self):
        def run():
            if self.torch is None:
                return {'status': 'skipped', 'reason': 'PyTorch environment unavailable'}
            verified, outputs_checked = 0, 0
            for tokens in (1, 8, 9, 32, 128, 256):
                inputs = self.inputs(tokens)
                fns = self.functions(inputs)
                for mode in MODES:
                    for reverse in (False, True):
                        ref = self.load_inputs(inputs, tokens, mode, seed=604 if not reverse else 967, reverse_mapping=reverse)
                        for stage in STAGES:
                            self.check(fns[stage](), ref[stage])
                            outputs_checked += 1
                        self.check(fns['annotated_control'](), ref['full_diagnostic'])
                        verified += 1
            return {'status': 'passed', 'device': 'cuda:0' if self.gpu else 'cpu', 'route_cases': verified,
                    'stage_checks': outputs_checked, 'annotated_control_checks': verified,
                    'note': 'Includes sentinel, all-remote, all-local, hot-local and uniform cases, reversed expert mapping, and 64/72-route boundary. Boundary is coverage only; V4 alignment is not executed.'}
        return self.section('correctness', run)

    def capture(self, fn):
        t = self.torch
        start = time.perf_counter()
        stream = t.cuda.Stream()
        stream.wait_stream(t.cuda.current_stream())
        with t.cuda.stream(stream):
            for _ in range(10):
                fn()
        t.cuda.current_stream().wait_stream(stream)
        t.cuda.synchronize()
        graph = t.cuda.CUDAGraph()
        with t.cuda.graph(graph):
            output = fn()
        t.cuda.synchronize()
        return graph, output, (time.perf_counter() - start) * 1000

    def measure_variants(self, variants, identity):
        t = self.torch
        for fn in variants.values():
            for _ in range(20):
                fn()
        a, b = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
        a.record()
        b.record()
        b.synchronize()  # Materialize lazy event objects outside timed samples.
        collected = defaultdict(list)
        rng = random.Random(741)
        for sample in range(self.samples):
            names = list(variants)
            rng.shuffle(names)
            for position, name in enumerate(names):
                fn = variants[name]
                t.cuda.synchronize()
                start = time.perf_counter_ns()
                a.record()
                for _ in range(self.repeats):
                    fn()
                b.record()
                b.synchronize()
                wall = (time.perf_counter_ns() - start) / 1000 / self.repeats
                event = a.elapsed_time(b) * 1000 / self.repeats
                row = dict(identity, variant=name, sample=sample, order_in_sample=position,
                           event_us=event, synchronized_wall_us=wall, repeats=self.repeats)
                self.raw_samples.append(row)
                collected[name].append(row)
        for name, rows in collected.items():
            e = statistics_us([r['event_us'] for r in rows])
            w = statistics_us([r['synchronized_wall_us'] for r in rows])
            self.timings.append(dict(identity, variant=name, samples=self.samples, repeats=self.repeats,
                                     **{'event_us_' + k: v for k, v in e.items()},
                                     **{'wall_us_' + k: v for k, v in w.items()}))

    def benchmark(self):
        def run():
            if not self.gpu:
                return {'status': 'skipped', 'reason': 'No active CUDA GPU; CPU timings are not substituted'}
            if self.report['sections'].get('correctness', {}).get('status') != 'passed':
                return {'status': 'skipped', 'reason': 'Correctness gate did not pass'}
            for tokens in self.tokens:
                for pattern in ('uniform', 'hot_local'):
                    for stage in STAGES:
                        identity = {'tokens': tokens, 'routes': tokens * 8, 'pattern': pattern, 'stage': stage}
                        graph = graph_output = None
                        try:
                            inputs = self.inputs(tokens)
                            base_ref = self.load_inputs(inputs, tokens, pattern)
                            fns = self.functions(inputs)
                            fn = fns[stage]
                            self.check(fn(), base_ref[stage])
                            graph, graph_output, capture_ms = self.capture(fn)
                            for j, mode in enumerate(MODES):
                                ref = self.load_inputs(inputs, tokens, mode, seed=1201+j, reverse_mapping=bool(j % 2))
                                graph.replay()
                                self.check(graph_output, ref[stage])
                            base_ref = self.load_inputs(inputs, tokens, pattern)
                            graph.replay()
                            self.check(graph_output, base_ref[stage])
                            variants = {'eager': fn, 'graph_replay': graph.replay}
                            if stage == 'full_diagnostic':
                                variants['eager_annotated_control'] = fns['annotated_control']
                            self.measure_variants(variants, identity)
                            self.check(fn(), base_ref[stage])
                            graph.replay()
                            self.check(graph_output, base_ref[stage])
                            self.case_results.append(dict(identity, status='passed', changed_input_replays=5, warmup_and_capture_ms=capture_ms))
                        except Exception as exc:
                            # A failed case must not leave publishable timing rows behind.
                            self.timings = [r for r in self.timings if not all(r[k] == v for k, v in identity.items())]
                            self.raw_samples = [r for r in self.raw_samples if not all(r[k] == v for k, v in identity.items())]
                            self.case_results.append(dict(identity, status='failed', error=repr(exc), traceback=traceback.format_exc(limit=6)))
                        finally:
                            if graph is not None:
                                self.torch.cuda.synchronize()
                            graph = graph_output = None
                    self.save()
                    print('Completed', tokens, 'tokens,', pattern, flush=True)
            failed = sum(r['status'] != 'passed' for r in self.case_results)
            return {'status': 'passed' if not failed else 'partial', 'cases': len(self.case_results),
                    'passed_cases': len(self.case_results)-failed, 'failed_cases': failed,
                    'timing_rows': len(self.timings), 'raw_samples': len(self.raw_samples),
                    'note': 'Each row is one isolated stage or the full pipeline. Do not sum stage timings. Graph replay changes host dispatch and allocation behavior together; the difference is not pure launch cost.'}
        return self.section('benchmark', run)

    def profile(self):
        def run():
            if not self.gpu or self.report['sections'].get('correctness', {}).get('status') != 'passed':
                return {'status': 'skipped', 'reason': 'Requires validated CUDA diagnostic'}
            t = self.torch
            inputs = self.inputs(128)
            reference = self.load_inputs(inputs, 128, 'hot_local')
            fns = self.functions(inputs)
            for _ in range(10):
                fns['annotated_control']()
            t.cuda.synchronize()
            with t.profiler.profile(activities=[t.profiler.ProfilerActivity.CPU, t.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=False) as prof:
                for _ in range(6):
                    output = fns['annotated_control']()
                    prof.step()
                t.cuda.synchronize()
            self.check(output, reference['full_diagnostic'])
            path = self.out / 'routing_trace.json'
            prof.export_chrome_trace(str(path))
            events = json.loads(path.read_text())['traceEvents']
            # Explicitly exclude annotation spans and operator aggregates from GPU work counts.
            device = [e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
            ranges = [e for e in events if e.get('cat') == 'user_annotation' and e.get('name', '').startswith('diagnostic/')]
            launches = {e['args']['correlation']: e for e in events if e.get('cat') == 'cuda_runtime' and 'correlation' in e.get('args', {})}
            rows = []
            for e in device:
                launch = launches.get(e.get('args', {}).get('correlation'))
                matches = [i for i, a in enumerate(ranges) if launch and (a['pid'], a['tid']) == (launch['pid'], launch['tid']) and a['ts'] <= launch['ts'] < a['ts'] + a['dur']]
                index = matches[0] if len(matches) == 1 else None
                rows.append({'category': e['cat'], 'name': e['name'], 'duration_us': e.get('dur', 0),
                             'stage': ranges[index]['name'] if index is not None else 'unattributed',
                             'range_index': index, 'device': e.get('args', {}).get('device'), 'stream': e.get('args', {}).get('stream')})
            write_csv(self.out / 'trace_device_operations.csv', rows)
            stage_rows = []
            occurrences = Counter()
            for i, a in enumerate(ranges):
                children = [r for r in rows if r['range_index'] == i]
                occurrences[a['name']] += 1
                stage_rows.append({'stage': a['name'], 'iteration': occurrences[a['name']], 'cpu_range_us': a['dur'],
                                   'kernel_count': sum(r['category'] == 'kernel' for r in children),
                                   'kernel_sum_us': sum(r['duration_us'] for r in children if r['category'] == 'kernel'),
                                   'copy_memset_count': sum(r['category'] != 'kernel' for r in children),
                                   'copy_memset_sum_us': sum(r['duration_us'] for r in children if r['category'] != 'kernel')})
            write_csv(self.out / 'trace_stages.csv', stage_rows)
            counts = Counter(e.get('cat', 'uncategorized') for e in events)
            unattributed = sum(r['stage'] == 'unattributed' for r in rows)
            return {'status': 'passed' if counts['kernel'] and not unattributed else 'partial', 'trace_categories': dict(counts),
                    'gpu_kernel_count': counts['kernel'], 'device_copy_count': counts['gpu_memcpy'], 'unattributed_device_operations': unattributed,
                    'note': 'Separate profiler-on eager run. Annotation spans are not kernels. CPU spans and GPU work overlap; do not add them. Empty device CSV means CUDA tracing was unavailable, not zero GPU cost.'}
        return self.section('profiler', run)

    def export(self):
        self.save()
        statuses = {k: v['status'] for k, v in self.report['sections'].items()}
        missing = [s for s in ('environment', 'correctness', 'benchmark', 'profiler') if s not in statuses]
        lines = ['NOTEBOOK 02: ROUTING OVERHEAD', 'Run: ' + self.report['run_id'], '',
                 self.report['scope'], '', 'Section status:']
        lines += [k + ': ' + v for k, v in statuses.items()]
        if missing:
            lines += ['Missing sections: ' + ', '.join(missing)]
        lines += ['', 'FULL DIAGNOSTIC: event median microseconds (p10-p90)']
        for row in self.timings:
            if row['stage'] == 'full_diagnostic':
                lines.append('{tokens:3d} tokens {pattern:9s} {variant:24s}: {event_us_median:.2f} ({event_us_p10:.2f}-{event_us_p90:.2f})'.format(**row))
        lines += ['', 'READING THESE RESULTS',
                  '- eager: ordinary Python dispatch with labels disabled.',
                  '- graph_replay: same operator sequence using captured storage and GPU graph replay.',
                  '- eager_annotated_control: same eager work plus record_function labels; no active profiler.',
                  '- Input generation/copies, graph capture, correctness checks and trace collection are outside timed samples.',
                  '- timings.csv includes isolated conversion, exact project mapping, synthetic sort/count, and padding counts.',
                  '- raw_samples.csv records randomized within-sample variant order, event intervals and synchronized wall time.',
                  '- Event intervals may include gaps between kernels. Graph deltas combine host/dispatch/allocation changes.',
                  '- Isolated stage times are not additive. These are warm repeated-buffer measurements, not single-request latency.',
                  '- Mapping helper is copied verbatim from the source hash in report.json; other stages are diagnostic code.',
                  '- Each graph was checked with changed routes and mapping values before measurement; failed cases publish no timings.',
                  '- Trace CSVs exclude annotation spans from kernel/copy counts. Inspect first trace iteration separately for profiler startup effects.',
                  '- Different package versions, clocks and hosts can change results; compare variants within this run first.',
                  '- No full alignment buffer, expert GEMM, attention, scheduler/tokenizer initialization or multi-GPU work is measured.',
                  '- No H200 speedup or serving percentage follows from these values.', '',
                  'Next: return this entire ZIP for analysis before selecting a serving change.',
                  'Method references: https://docs.pytorch.org/docs/stable/notes/cuda.html and https://docs.pytorch.org/docs/stable/profiler.html', '']
        summary = '\n'.join(lines)
        (self.out / 'SUMMARY.txt').write_text(summary, encoding='utf-8')
        (self.out / 'summary.html').write_text('<!doctype html><meta charset="utf-8"><title>Routing overhead</title><style>body{font:16px system-ui;margin:32px}pre{white-space:pre-wrap}</style><pre>' + html.escape(summary) + '</pre>', encoding='utf-8')
        checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(self.out.iterdir()) if p.is_file() and p.name != 'checksums.json'}
        (self.out / 'checksums.json').write_text(json.dumps(checksums, indent=2), encoding='utf-8')
        archive = self.out.with_suffix('.zip')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(self.out.iterdir()):
                if p.is_file():
                    z.write(p, p.name)
        print(summary)
        print('Download:', archive)
        return archive
