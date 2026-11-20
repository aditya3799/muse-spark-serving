"""One mapping candidate, independent correctness, and repeated within-run A/B tests."""
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


def routes(tokens, pattern, seed=604):
    rng = random.Random(seed)
    values = []
    remote = [x for x in range(256) if not 64 <= x < 96]
    for _ in range(tokens):
        if pattern == 'all_local':
            row = rng.sample(range(64, 96), 8)
        elif pattern == 'all_remote':
            row = rng.sample(remote, 8)
        elif pattern == 'hot_local':
            row = [64, 65, 66, 67] + rng.sample(remote, 4)
        elif pattern == 'all_negative':
            row = [-1, -2, -2147483648, -3, -1, -7, -10, -99]
        else:
            row = rng.sample(range(256), 8)
            if pattern == 'sentinel':
                row[:3] = [-1, -2, -2147483648]
        values.extend(row)
    return values


def expert_mapping(kind='contiguous', seed=604):
    result = [-1] * 256
    if kind == 'contiguous':
        result[64:96] = range(32)
    elif kind == 'permuted':
        rng = random.Random(seed)
        indices, local = rng.sample(range(256), 32), list(range(32))
        rng.shuffle(local)
        for i, value in zip(indices, local):
            result[i] = value
    elif kind == 'shared':
        result[64:96] = range(32)
        result[248:] = range(32, 40)
    elif kind != 'all_remote':
        raise ValueError(kind)
    return result


def reference(values, mapping):
    # No tensor operations and no shifted table: directly specify the desired mapping.
    assert all(x < len(mapping) for x in values)
    return [-1 if x < 0 else mapping[x] for x in values]


def write_csv(path, rows):
    if rows:
        with Path(path).open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)


def stats(values):
    s = sorted(values)
    return {'median': statistics.median(s), 'p10': s[int(.1*(len(s)-1))], 'p90': s[int(.9*(len(s)-1))]}


class MappingStudy:
    def __init__(self, manifest, helper_source, output_root=None, force_cpu=False,
                 samples=11, repeats=20, trials=3, tokens=(9, 32, 128, 256)):
        assert hashlib.sha256(helper_source.encode()).hexdigest() == manifest['mapping_helper_sha256']
        assert samples >= 5 and repeats >= 1 and trials >= 1
        self.manifest, self.helper_source = manifest, helper_source
        self.samples, self.repeats, self.trials, self.tokens = samples, repeats, trials, tuple(tokens)
        self.t, self.gpu, self.force_cpu = None, False, force_cpu
        root = Path(output_root) if output_root else (Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd())
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:6]
        self.out = root / ('v4_mapping_candidate_' + run_id)
        self.out.mkdir(parents=True, exist_ok=False)
        self.report = {'schema_version': 3, 'run_id': run_id, 'source_manifest': manifest,
            'scope': 'Exact fallback mapping helper versus a shifted-lookup candidate. No serving changes or H200 prediction.',
            'config': {'tokens': list(tokens), 'patterns': ['uniform', 'hot_local'], 'dtypes': ['int32', 'int64'], 'scopes': ['helper_only', 'fallback_cast_map'], 'samples': samples, 'repeats': repeats, 'trials': trials, 'lookup_bytes': 257*4},
            'candidate': 'lookup = cat([-1], mapping); result = lookup[ids.clamp_min(-1) + 1]. Every negative ID maps to lookup[0]. Nonnegative IDs read the original mapping entry. Positive IDs must be in bounds.',
            'method': 'Interleaved within-sample variants, shuffled cases per independent trial. Warm repeated-buffer CUDA-event and synchronized-wall timings. Same output allocation contract; graph-private pools replace eager allocator work. No profiler or labels in timed calls. Cached candidate excludes lookup preparation; rebuilt candidate includes it; setup and in-place refresh are measured separately.',
            'sections': {}}
        self.rows, self.raw, self.cases = [], [], []
        (self.out / 'mapping_helper.py').write_text(helper_source, encoding='utf-8')
        self.save()

    def save(self):
        (self.out / 'report.json').write_text(json.dumps(self.report, indent=2), encoding='utf-8')
        for name, rows in [('timings.csv', self.rows), ('raw_samples.csv', self.raw), ('benchmark_cases.csv', self.cases)]:
            write_csv(self.out / name, rows)

    def section(self, name, fn):
        begin = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:
            result = {'status': 'failed', 'error': repr(exc), 'traceback': traceback.format_exc(limit=8)}
        result['wall_seconds'] = round(time.perf_counter()-begin, 3)
        self.report['sections'][name] = result
        self.save()
        print(name + ': ' + result['status'], flush=True)
        print(json.dumps(result, indent=2))
        return result

    def environment(self):
        def run():
            import torch
            self.t = torch
            namespace = {'torch': torch}
            exec(self.helper_source, namespace)
            self.baseline = namespace['_map_topk_ids_to_local_experts']
            self.gpu = torch.cuda.is_available() and not self.force_cpu
            if self.gpu:
                torch.cuda.set_device(0)
            packages = {}
            for name in ('torch', 'triton', 'numpy', 'ipykernel'):
                try:
                    packages[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    packages[name] = None
            result = {'status': 'passed' if self.gpu else 'cpu_only', 'python': sys.version, 'os': platform.system(), 'packages': packages, 'cuda_version': torch.version.cuda, 'gpu_devices': [], 'active_device': 0 if self.gpu else None}
            if self.gpu:
                for i in range(torch.cuda.device_count()):
                    p = torch.cuda.get_device_properties(i)
                    result['gpu_devices'].append({'index': i, 'name': p.name, 'capability': [p.major, p.minor], 'memory_bytes': p.total_memory})
                try:
                    smi = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,pstate,power.draw,clocks.sm', '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
                    result['nvidia_smi'] = smi.stdout.strip()
                except Exception as exc:
                    result['nvidia_smi_note'] = repr(exc)
            else:
                result['note'] = 'CPU correctness only. Select GPU T4, restart and Run All for GPU measurements.'
            return result
        return self.section('environment', run)

    def build_lookup(self, mapping):
        return self.t.cat((mapping.new_full((1,), -1), mapping))

    def candidate(self, ids, lookup):
        return lookup[ids.clamp_min(-1) + 1]

    def allocate(self, tokens, dtype, strided=False):
        t = self.t
        device = 'cuda' if self.gpu else 'cpu'
        raw = t.empty((tokens, 16 if strided else 8), dtype=getattr(t, dtype), device=device)
        raw = raw[:, ::2] if strided else raw
        return {'raw': raw, 'mapping': t.empty(256, dtype=t.int32, device=device), 'lookup': t.empty(257, dtype=t.int32, device=device)}

    def load(self, inputs, tokens, pattern='uniform', kind='contiguous', seed=604):
        t = self.t
        values, mapping = routes(tokens, pattern, seed), expert_mapping(kind, seed)
        inputs['raw'].copy_(t.tensor(values, dtype=inputs['raw'].dtype).reshape(tokens, 8))
        inputs['mapping'].copy_(t.tensor(mapping, dtype=t.int32))
        # Update the captured lookup storage in place; never replace its pointer.
        inputs['lookup'].copy_(self.build_lookup(inputs['mapping']))
        return reference(values, mapping)

    def functions(self, inputs, scope):
        raw, mapping, lookup = (inputs[k] for k in ('raw', 'mapping', 'lookup'))
        if scope == 'helper_only':
            return {'baseline': lambda: self.baseline(raw, mapping),
                    'cached_lookup': lambda: self.candidate(raw, lookup),
                    'rebuilt_lookup': lambda: self.candidate(raw, self.build_lookup(mapping))}
        def baseline():
            return self.baseline(raw.to(self.t.int32), mapping).contiguous()
        def cached():
            return self.candidate(raw.to(self.t.int32), lookup).contiguous()
        def rebuilt():
            return self.candidate(raw.to(self.t.int32), self.build_lookup(mapping)).contiguous()
        return {'baseline': baseline, 'cached_lookup': cached, 'rebuilt_lookup': rebuilt}

    def check(self, got, expected, tokens):
        assert tuple(got.shape) == (tokens, 8), 'Shape changed'
        assert got.dtype == self.t.int32, 'Output dtype changed'
        assert got.device.type == ('cuda' if self.gpu else 'cpu'), 'Output device changed'
        assert got.is_contiguous(), 'Output is not contiguous'
        assert got.flatten().tolist() == expected, 'Independent CPU reference mismatch'

    def correctness(self):
        def run():
            if self.t is None:
                return {'status': 'skipped', 'reason': 'PyTorch unavailable'}
            cases = checks = 0
            for tokens in (0, 1, 8, 9, 32, 128, 256):
                for dtype in ('int32', 'int64'):
                    for strided in (False, True):
                        inputs = self.allocate(tokens, dtype, strided)
                        for pattern in ('uniform', 'hot_local', 'all_local', 'all_remote', 'sentinel', 'all_negative'):
                            for kind in ('contiguous', 'permuted', 'shared', 'all_remote'):
                                expected = self.load(inputs, tokens, pattern, kind)
                                before_raw, before_map = inputs['raw'].clone(), inputs['mapping'].clone()
                                for scope in ('helper_only', 'fallback_cast_map'):
                                    for fn in self.functions(inputs, scope).values():
                                        self.check(fn(), expected, tokens)
                                        checks += 1
                                assert self.t.equal(inputs['raw'], before_raw) and self.t.equal(inputs['mapping'], before_map), 'Input modified'
                                cases += 1
            return {'status': 'passed', 'device': 'cuda:0' if self.gpu else 'cpu', 'input_cases': cases, 'output_checks': checks,
                    'coverage': 'Empty/1/8/9/32/128/256 tokens, int32/int64, contiguous/strided inputs, six route patterns, four mappings, negative IDs down to int32 minimum. Nonnegative out-of-range IDs excluded, as required by the original helper. Input values must survive the serving int64-to-int32 cast.'}
        return self.section('correctness', run)

    def capture(self, fn):
        t = self.t
        start = time.perf_counter()
        stream = t.cuda.Stream()
        stream.wait_stream(t.cuda.current_stream())
        with t.cuda.stream(stream):
            for _ in range(5):
                fn()
        t.cuda.current_stream().wait_stream(stream)
        t.cuda.synchronize()
        graph = t.cuda.CUDAGraph()
        with t.cuda.graph(graph):
            output = fn()
        t.cuda.synchronize()
        return graph, output, (time.perf_counter()-start)*1000

    def measure(self, variants, identity):
        t = self.t
        for fn in variants.values():
            for _ in range(10):
                fn()
        a, b = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
        a.record(); b.record(); b.synchronize()
        rng = random.Random(731 + identity['trial'])
        grouped = defaultdict(list)
        for sample in range(self.samples):
            order = list(variants)
            rng.shuffle(order)
            for position, name in enumerate(order):
                fn = variants[name]
                t.cuda.synchronize()
                begin = time.perf_counter_ns()
                a.record()
                for _ in range(self.repeats):
                    fn()
                b.record(); b.synchronize()
                row = dict(identity, variant=name, sample=sample, order_in_sample=position, repeats=self.repeats,
                           wall_us=(time.perf_counter_ns()-begin)/1000/self.repeats, event_us=a.elapsed_time(b)*1000/self.repeats)
                self.raw.append(row)
                grouped[name].append(row)
        for name, rows in grouped.items():
            self.rows.append(dict(identity, variant=name, samples=self.samples, repeats=self.repeats,
                **{'event_us_'+k: v for k,v in stats([r['event_us'] for r in rows]).items()},
                **{'wall_us_'+k: v for k,v in stats([r['wall_us'] for r in rows]).items()}))

    def benchmark(self):
        def run():
            if not self.gpu or self.report['sections'].get('correctness', {}).get('status') != 'passed':
                return {'status': 'skipped', 'reason': 'Requires validated CUDA correctness'}
            configurations = [(n, d, p, s) for n in self.tokens for d in ('int32', 'int64') for p in ('uniform', 'hot_local') for s in ('helper_only', 'fallback_cast_map')]
            for trial in range(self.trials):
                order = list(configurations)
                random.Random(902+trial).shuffle(order)
                for index, (tokens, dtype, pattern, scope) in enumerate(order):
                    identity = {'trial': trial, 'case_order': index, 'tokens': tokens, 'dtype': dtype, 'pattern': pattern, 'scope': scope}
                    captures, variants = {}, {}
                    try:
                        inputs = self.allocate(tokens, dtype)
                        expected = self.load(inputs, tokens, pattern)
                        fns = self.functions(inputs, scope)
                        for name, fn in fns.items():
                            self.check(fn(), expected, tokens)
                            captures[name] = self.capture(fn)
                        for pattern_check, kind in [('all_negative','contiguous'), ('sentinel','shared'), ('uniform','permuted'), ('all_local','all_remote')]:
                            changed = self.load(inputs, tokens, pattern_check, kind, 1901+trial)
                            for graph, output, _ in captures.values():
                                graph.replay()
                                self.check(output, changed, tokens)
                        expected = self.load(inputs, tokens, pattern)
                        for name, fn in fns.items():
                            graph, output, _ = captures[name]
                            graph.replay(); self.check(output, expected, tokens)
                            variants[name+'_eager'] = fn
                            variants[name+'_graph'] = graph.replay
                        self.measure(variants, identity)
                        for name, fn in fns.items():
                            self.check(fn(), expected, tokens)
                            graph, output, _ = captures[name]
                            graph.replay(); self.check(output, expected, tokens)
                        self.cases.append(dict(identity, status='passed', changed_input_graph_checks=12,
                                               capture_setup_ms=sum(x[2] for x in captures.values())))
                    except Exception as exc:
                        self.rows = [r for r in self.rows if not all(r[k] == v for k,v in identity.items())]
                        self.raw = [r for r in self.raw if not all(r[k] == v for k,v in identity.items())]
                        self.cases.append(dict(identity, status='failed', error=repr(exc), traceback=traceback.format_exc(limit=6)))
                    finally:
                        self.t.cuda.synchronize()
                        variants.clear(); captures.clear()
                    if (index+1) % 8 == 0:
                        self.save()
                        print('Trial', trial+1, ':', index+1, '/', len(order), 'cases', flush=True)
                self.save()
            failed = sum(c['status'] != 'passed' for c in self.cases)
            return {'status': 'passed' if not failed else 'partial', 'cases': len(self.cases), 'failed_cases': failed,
                    'changed_input_graph_checks': sum(c.get('changed_input_graph_checks', 0) for c in self.cases),
                    'timing_rows': len(self.rows), 'raw_samples': len(self.raw)}
        return self.section('benchmark', run)

    def setup_cost(self):
        def run():
            if not self.gpu or self.report['sections'].get('correctness', {}).get('status') != 'passed':
                return {'status': 'skipped', 'reason': 'Requires validated CUDA correctness'}
            inputs = self.allocate(9, 'int32')
            self.load(inputs, 9)
            mapping, lookup = inputs['mapping'], inputs['lookup']
            fns = {'build_table': lambda: self.build_lookup(mapping),
                   'refresh_existing_table': lambda: lookup.copy_(self.build_lookup(mapping))}
            captures = {name: self.capture(fn) for name, fn in fns.items()}
            for kind in ('shared', 'permuted', 'all_remote'):
                self.load(inputs, 9, kind=kind)
                expected = [-1] + expert_mapping(kind)
                for graph, output, _ in captures.values():
                    graph.replay()
                    assert output.tolist() == expected
            self.load(inputs, 9)
            variants = {}
            for name, fn in fns.items():
                variants[name+'_eager'] = fn
                variants[name+'_graph'] = captures[name][0].replay
            for trial in range(self.trials):
                self.measure(variants, {'trial': trial, 'case_order': -1, 'tokens': 0, 'dtype': 'int32', 'pattern': 'table_setup', 'scope': 'lookup_setup'})
            return {'status': 'passed', 'persistent_lookup_bytes': lookup.numel()*lookup.element_size(),
                    'changed_mapping_graph_checks': 6, 'note': 'Build = create a new 257-entry table. Refresh = build then copy into existing captured storage. Cached candidate excludes both costs. Rebuilt candidate in the main benchmark includes table creation on every call.'}
        return self.section('setup_cost', run)

    def profile(self):
        def run():
            if not self.gpu or self.report['sections'].get('correctness', {}).get('status') != 'passed':
                return {'status': 'skipped', 'reason': 'Requires validated CUDA correctness'}
            t = self.t
            inputs = self.allocate(128, 'int64')
            expected = self.load(inputs, 128, 'hot_local')
            fns = self.functions(inputs, 'fallback_cast_map')
            for fn in fns.values():
                for _ in range(10): fn()
            t.cuda.synchronize()
            with t.profiler.profile(activities=[t.profiler.ProfilerActivity.CPU, t.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=False) as prof:
                for iteration in range(6):
                    names = list(fns)
                    random.Random(30+iteration).shuffle(names)
                    for name in names:
                        with t.profiler.record_function('mapping/'+name):
                            output = fns[name]()
                    prof.step()
                t.cuda.synchronize()
            self.check(output, expected, 128)
            path = self.out / 'mapping_trace.json'
            prof.export_chrome_trace(str(path))
            return parse_trace(path, self.out)
        return self.section('profiler', run)

    def export(self):
        self.save()
        comparisons = []
        for case in self.cases:
            if case['status'] != 'passed': continue
            key = {k: case[k] for k in ('trial', 'tokens', 'dtype', 'pattern', 'scope')}
            matches = {r['variant']: r for r in self.rows if all(r[k] == v for k,v in key.items())}
            for mode in ('eager', 'graph'):
                for candidate in ('cached_lookup', 'rebuilt_lookup'):
                    baseline, tested = matches['baseline_'+mode], matches[candidate+'_'+mode]
                    comparisons.append(dict(key, execution=mode, candidate=candidate,
                        baseline_event_us=baseline['event_us_median'], candidate_event_us=tested['event_us_median'],
                        baseline_over_candidate=baseline['event_us_median']/tested['event_us_median'],
                        event_saved_us=baseline['event_us_median']-tested['event_us_median'],
                        wall_saved_us=baseline['wall_us_median']-tested['wall_us_median']))
        write_csv(self.out/'comparisons.csv', comparisons)
        lines = ['NOTEBOOK 03: MAPPING CANDIDATE', 'Run: '+self.report['run_id'], '', self.report['scope'], '']
        for name in ('environment', 'correctness', 'benchmark', 'setup_cost', 'profiler'):
            lines.append(name+': '+self.report['sections'].get(name, {}).get('status', 'not_run'))
        lines += ['', 'CACHED CANDIDATE, FALLBACK CAST+MAP, INT64 / HOT-LOCAL', 'Trial Tokens Mode   Baseline us Candidate us Ratio (baseline/candidate)']
        for r in comparisons:
            if r['candidate']=='cached_lookup' and r['scope']=='fallback_cast_map' and r['dtype']=='int64' and r['pattern']=='hot_local':
                lines.append('{trial:5d} {tokens:6d} {execution:6s} {baseline_event_us:11.2f} {candidate_event_us:12.2f} {baseline_over_candidate:8.3f}'.format(**r))
        lines += ['', 'INTERPRETATION',
            '- Ratio > 1 favors the candidate; ratio < 1 favors baseline. Do not accept a candidate merely because one row wins.',
            '- Inspect all three trials, p10-p90 sample spreads and synchronized wall times, especially graph execution.',
            '- Cached lookup requires a reusable, valid table. Extra storage: 1,028 bytes per mapping for int32 entries.',
            '- Refresh the table when the source mapping changes, and update the existing storage before replay. Main cached timings exclude this refresh; setup_cost measures it separately.',
            '- Rebuilt lookup includes table construction each call; it answers whether any benefit survives lack of caching.',
            '- Candidate clamp_min(-1) preserves every negative ID, including negatives below -1. Nonnegative IDs must be valid mapping indices.',
            '- helper_only includes int32 or int64 inputs directly. fallback_cast_map performs the same int32 cast and contiguous output step as the inspected serving fallback.',
            '- Positive IDs and negative test IDs fit int32; this notebook does not authorize narrowing arbitrary int64 indices.',
            '- Actual V4 already fuses mapping into eligible <=64-route alignment. This experiment targets the standalone fallback and leaves its threshold unchanged.',
            '- Timings exclude input generation/copies, capture and correctness checks. Timed operations contain no labels/profiler.',
            '- Graph replay changes host dispatch and allocator behavior. Event intervals may contain gaps; they are not summed kernel duration.',
            '- Trace is a separate instrumented run, with only kernel/copy/memset counted as device operations. No serving latency fractions.',
            '- No installs, model downloads, Triton compilation, serving modifications, or H200 speedup claims.', '',
            'Return this entire ZIP for review. Failure and skipped statuses are evidence, not zero latency.', '']
        text = '\n'.join(lines)
        (self.out/'SUMMARY.txt').write_text(text, encoding='utf-8')
        (self.out/'summary.html').write_text('<!doctype html><meta charset="utf-8"><title>Mapping experiment</title><style>body{font:16px system-ui;margin:32px}pre{white-space:pre-wrap}</style><pre>'+html.escape(text)+'</pre>', encoding='utf-8')
        checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(self.out.iterdir()) if p.is_file() and p.name != 'checksums.json'}
        (self.out/'checksums.json').write_text(json.dumps(checksums, indent=2), encoding='utf-8')
        archive = self.out.with_suffix('.zip')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(self.out.iterdir()):
                if p.is_file(): z.write(p, p.name)
        print(text)
        print('Download:', archive)
        return archive


def parse_trace(path, out):
    events = json.loads(Path(path).read_text())['traceEvents']
    ranges = [e for e in events if e.get('cat') == 'user_annotation' and e.get('name','').startswith('mapping/')]
    launches = {e['args']['correlation']: e for e in events if e.get('cat') == 'cuda_runtime' and 'correlation' in e.get('args', {})}
    rows = []
    for e in events:
        if e.get('cat') not in ('kernel', 'gpu_memcpy', 'gpu_memset'): continue
        launch = launches.get(e.get('args', {}).get('correlation'))
        matches = [i for i,a in enumerate(ranges) if launch and (a['pid'],a['tid']) == (launch['pid'],launch['tid']) and a['ts'] <= launch['ts'] < a['ts']+a['dur']]
        index = matches[0] if len(matches)==1 else None
        rows.append({'category': e['cat'], 'name': e['name'], 'duration_us': e.get('dur',0), 'variant': ranges[index]['name'] if index is not None else 'unattributed', 'range_index': index})
    write_csv(Path(out)/'trace_device_operations.csv', rows)
    counts, summaries = Counter(), []
    for i,a in enumerate(ranges):
        counts[a['name']] += 1
        children = [r for r in rows if r['range_index']==i]
        summaries.append({'variant': a['name'], 'iteration': counts[a['name']], 'cpu_range_us': a['dur'],
            'kernel_count': sum(r['category']=='kernel' for r in children),
            'kernel_sum_us': sum(r['duration_us'] for r in children if r['category']=='kernel'),
            'copy_memset_count': sum(r['category']!='kernel' for r in children),
            'copy_memset_sum_us': sum(r['duration_us'] for r in children if r['category']!='kernel')})
    write_csv(Path(out)/'trace_variants.csv', summaries)
    cats = Counter(e.get('cat','uncategorized') for e in events)
    unknown = sum(r['variant']=='unattributed' for r in rows)
    return {'status': 'passed' if cats['kernel'] and not unknown else 'partial', 'trace_categories': dict(cats),
            'unattributed_device_operations': unknown, 'note': 'Only actual kernels/copies/memsets counted. Annotation ranges excluded. CPU-only trace is partial, not zero GPU cost. Separate profiler-on eager execution; do not subtract its durations from benchmark timings.'}
