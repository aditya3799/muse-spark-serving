"""Timing utilities extracted from validated Notebook 03; no candidate code."""
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

class ExperimentCore:
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

    def check(self, got, expected, tokens):
        assert tuple(got.shape) == (tokens, 8), 'Shape changed'
        assert got.dtype == self.t.int32, 'Output dtype changed'
        assert got.device.type == ('cuda' if self.gpu else 'cpu'), 'Output device changed'
        assert got.is_contiguous(), 'Output is not contiguous'
        assert got.flatten().tolist() == expected, 'Independent CPU reference mismatch'

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
