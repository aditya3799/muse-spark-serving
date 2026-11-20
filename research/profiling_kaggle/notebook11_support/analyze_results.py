import csv,hashlib,io,json,statistics,zipfile
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\OpenKernelChallenge\serving\profiling_kaggle')
archive=next(ROOT.rglob('v4_cache_choice_20261114T183520Z_9b0849.zip'))
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==len(set(z.namelist()))
    hashes=json.loads(z.read('checksums.json'));assert set(hashes)==set(z.namelist())-{'checksums.json'}
    for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    report=json.loads(z.read('report.json'))
    def table(n):return list(csv.DictReader(io.StringIO(z.read(n).decode())))
    timings,raw,cases,comparisons,trace,devices=map(table,('timings.csv','raw_samples.csv','benchmark_cases.csv','comparisons.csv','trace_variants.csv','trace_device_operations.csv'))
assert report['source_manifest']==json.loads((ROOT/'notebook11_support/source_manifest_11.json').read_text())
assert all(r['status']=='passed' for r in report['sections'].values())
assert (len(timings),len(raw),len(cases),len(comparisons))==(288,3168,48,288)
assert all(c['status']=='passed' for c in cases)
assert sum(int(c['changed_input_output_checks']) for c in cases)==2592
keys=('trial','tokens','pattern','scope','stages');samples=defaultdict(list);lookup={}
for r in raw:samples[tuple(r[k] for k in keys)+(r['variant'],)].append(r)
for r in timings:
    key=tuple(r[k] for k in keys)+(r['variant'],);assert key not in lookup;lookup[key]=r;s=samples[key]
    assert len(s)==11 and len({x['sample'] for x in s})==11
    for metric in ('event','wall','enqueue'):
        assert all(float(x[metric+'_us'])>0 for x in s)
        assert abs(statistics.median(float(x[metric+'_us']) for x in s)-float(r[metric+'_us_median']))<1e-8
for r in comparisons:
    key=tuple(r[k] for k in keys);depth=int(r['stages'])
    for metric in ('event','wall','enqueue'):
        b=float(lookup[key+(r['baseline'],)][metric+'_us_median']);c=float(lookup[key+(r['candidate'],)][metric+'_us_median'])
        assert abs(float(r[metric+'_baseline_us'])-b)<1e-8
        assert abs(float(r[metric+'_candidate_us'])-c)<1e-8
        assert abs(float(r[metric+'_reduction_percent'])-100*(1-c/b))<1e-8
        assert abs(float(r[metric+'_saved_us'])-(b-c))<1e-8
        assert abs(float(r[metric+'_saved_us_per_stage'])-(b-c)/depth)<1e-8
assert len(devices)==672 and all(d['category']=='kernel' and d['variant']!='unattributed' for d in devices)
assert len(trace)==18
for r in trace:assert (int(r['kernel_count']),int(r['copy_memset_count']))==(48 if '/baseline_' in r['variant'] else 32,0)
def summary(rows,metric):
    vals=[float(r[metric+'_reduction_percent']) for r in rows]
    return {'wins':sum(v>0 for v in vals),'ties':sum(v==0 for v in vals),'comparisons':len(vals),'median_reduction_percent':statistics.median(vals),'min_reduction_percent':min(vals),'max_reduction_percent':max(vals),'median_saved_us_per_stage':statistics.median(float(r[metric+'_saved_us_per_stage']) for r in rows)}
overall=[];by_tokens=[]
for candidate,baseline in sorted(set((r['candidate'],r['baseline']) for r in comparisons)):
    selected=[r for r in comparisons if (r['candidate'],r['baseline'])==(candidate,baseline)]
    overall.append(dict(candidate=candidate,baseline=baseline,**{m:summary(selected,m) for m in ('event','wall','enqueue')}))
    for n in (9,32,128,512):by_tokens.append(dict(candidate=candidate,baseline=baseline,tokens=n,**{m:summary([r for r in selected if r['tokens']==str(n)],m) for m in ('event','wall','enqueue')}))
analysis={'run_id':report['run_id'],'zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'member_checksums_verified':len(hashes),'timing_rows_recomputed_all_three_metrics':288,'comparison_rows_verified':288,'overall':overall,'by_tokens':by_tokens,
    'decision':'Select the existing two-buffer draft for further static-map correctness validation. It preserves the registered map, avoids property-view eager overhead, and has no meaningful measured graph advantage to give up. This is not production promotion: explicit refresh remains required after every map edit.'}
lines=['NOTEBOOK 11: VERIFIED CACHE-CHOICE REVIEW','Run: '+report['run_id'],'',
    'VALIDATION',f'All {len(hashes)} member checksums and source provenance match. Recomputed all metrics for 288 rows from 3,168 samples and verified 288 comparisons. All 48 cases, 2,592 changed-input output checks and lifecycle/input/preservation gates passed.',
    'Trace: 672 kernels, no copies, no unattributed operations. Eight-stage baseline uses 48 kernels; both candidates use 32, in eager and graph schedules.','',
    'MATCHED RESULTS',
    'Candidate / Reference | Event reduction (wins/48) | Wall reduction | Saved event us per stage']
for r in overall:lines.append(f"{r['candidate']} / {r['baseline']} | {r['event']['median_reduction_percent']:.2f}% ({r['event']['wins']}/48) | {r['wall']['median_reduction_percent']:.2f}% | {r['event']['median_saved_us_per_stage']:.3f}")
lines+=['Negative reduction means slower. Values are descriptive matched medians, not confidence intervals or proof of statistical equivalence.','',
    'DECISION',
    'Prefer the existing two-buffer draft for the static-map path: 30.06% median lower event latency in eager execution and 24.91% in graph replay versus baseline, winning all 48 comparisons in each mode.',
    'The single-owner design is 10.91% slower than two-buffer in eager execution (median 8.111 us more per stage). This is consistent with property/view host overhead. Their device operation counts match.',
    'Single-owner graph latency is 0.56% higher at the median, about 0.107 us per stage; it wins 10/48 comparisons. This small difference does not establish equivalence or a universal ranking, but provides no performance reason here to accept its changed registered-buffer contract.',
    'Two-buffer retains the original named _expert_map buffer at storage offset zero. Single-owner changes it to a property at offset one. Both remain unvalidated in the complete serving framework and real fused kernels.',
    'The preservation gate explicitly observed stale output after directly changing only the two-buffer map; refreshing lookup restored correct captured-graph output. Its checked update method also restored correctness. This lifecycle obligation is real, not waived by static timing wins.',
    'Notebooks 06-08 showed frequent eager updates can be expensive. The selection is conditional on static or sufficiently reused maps and correct refresh ordering. Never add a refresh to every forward by default.','',
    'NEXT: 12_kaggle_two_buffer_gate.ipynb',
    'Freeze the exact existing V4-relative two-buffer draft; no new algorithm or active source edits. Stress all eight contiguous expert partitions, empty/small/boundary/large inputs, two ID dtypes, strided IDs and negative sentinels.',
    'Test explicit refresh and checked updates against an independent reference, graph visibility with stable pointers, rejected malformed updates, and moving modules followed by fresh capture. The notebook is a correctness gate, not another latency claim.',
    'Small fused branches are checked only for dispatch and metadata. Actual alignment/GEMMs, offset-free map use in real fused kernels, full checkpoint/model loading, framework integrations, startup and H200/full-serving performance remain untested.',
    'The selected draft is copied into notebook12_support for reproducibility and checked against the frozen V4 source in an isolated temporary tree. It is NOT applied to active V4.','']
(HERE/'NOTEBOOK11_ANALYSIS.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
(HERE/'NOTEBOOK11_REVIEW.txt').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'verified_members':len(hashes),'overall':overall},indent=2))
