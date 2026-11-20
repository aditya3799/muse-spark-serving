import csv,hashlib,io,json,statistics,zipfile
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\OpenKernelChallenge\serving\profiling_kaggle')
archive=next(ROOT.rglob('v4_update_overhead_20261114T175507Z_dd80bf.zip'))
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==len(set(z.namelist()))
    hashes=json.loads(z.read('checksums.json'));assert set(hashes)==set(z.namelist())-{'checksums.json'}
    for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    report=json.loads(z.read('report.json'))
    def table(n):return list(csv.DictReader(io.StringIO(z.read(n).decode())))
    timings,raw,cases,comparisons,trace,devices=map(table,('timings.csv','raw_samples.csv','benchmark_cases.csv','comparisons.csv','trace_variants.csv','trace_device_operations.csv'))
assert report['source_manifest']==json.loads((ROOT/'notebook08_support/source_manifest_08.json').read_text())
assert all(r['status']=='passed' for r in report['sections'].values())
assert (len(timings),len(raw),len(cases),len(comparisons))==(168,1848,24,192)
assert all(c['status']=='passed' for c in cases)
assert sum(int(c['poisoned_state_checks']) for c in cases)==504
keys=('trial','tokens','dtype','pattern','scope','replays_per_cycle');samples=defaultdict(list);lookup={}
for r in raw:samples[tuple(r[k] for k in keys)+(r['variant'],)].append(r)
for r in timings:
    key=tuple(r[k] for k in keys)+(r['variant'],);assert key not in lookup;lookup[key]=r;s=samples[key]
    assert len(s)==11 and len({x['sample'] for x in s})==11
    for metric in ('event','wall','enqueue'):
        assert all(float(x[metric+'_us'])>0 for x in s)
        assert abs(statistics.median(float(x[metric+'_us']) for x in s)-float(r[metric+'_us_median']))<1e-8
for r in comparisons:
    key=tuple(r[k] for k in keys)
    for metric in ('event','wall','enqueue'):
        b=float(lookup[key+(r['baseline'],)][metric+'_us_median']);c=float(lookup[key+(r['candidate'],)][metric+'_us_median'])
        assert abs(float(r[metric+'_baseline_us'])-b)<1e-8
        assert abs(float(r[metric+'_candidate_us'])-c)<1e-8
        assert abs(float(r[metric+'_reduction_percent'])-100*(1-c/b))<1e-8
assert len(devices)==156 and sum(d['category']=='kernel' for d in devices)==136 and sum(d['category']=='gpu_memcpy' for d in devices)==20
assert all(d['variant']!='unattributed' for d in devices)
assert len(trace)==20
for r in trace:assert (int(r['kernel_count']),int(r['copy_memset_count']))==(8 if r['variant'].startswith('mapping/baseline') else 6,1)
def summarize(rows,metric):
    vals=[float(r[metric+'_reduction_percent']) for r in rows]
    return {'wins':sum(v>0 for v in vals),'comparisons':len(vals),'median_reduction_percent':statistics.median(vals),'min_reduction_percent':min(vals),'max_reduction_percent':max(vals)}
groups=[];overall=[]
for candidate,baseline in sorted(set((r['candidate'],r['baseline']) for r in comparisons)):
    selected=[r for r in comparisons if (r['candidate'],r['baseline'])==(candidate,baseline)]
    overall.append(dict(candidate=candidate,baseline=baseline,**{m:summarize(selected,m) for m in ('event','wall','enqueue')}))
    for k in (1,2,4,8):groups.append(dict(candidate=candidate,baseline=baseline,replays_per_cycle=k,**{m:summarize([r for r in selected if int(r['replays_per_cycle'])==k],m) for m in ('event','wall','enqueue')}))
analysis={'run_id':report['run_id'],'zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'member_checksums_verified':len(hashes),'timing_rows_recomputed_all_three_metrics':168,'comparison_rows_verified':192,'groups':groups,'overall':overall,
    'decision':'Host update-path overhead is supported by matched API controls. Retained-view copy is a diagnostic win, not a production-ready API. Stop chasing per-call refresh overhead without evidence of production updates; next test static mapping across graph capture boundaries.'}
lines=['NOTEBOOK 08: VERIFIED UPDATE-OVERHEAD REVIEW','Run: '+report['run_id'],'',
    'VALIDATION',f'All {len(hashes)} ZIP member checksums and source provenance match. Recomputed all three metrics for 168 timing rows from 1,848 samples; verified 192 comparisons. 24 benchmark cases, 504 poisoned-state checks and inherited 576 output checks passed.',
    'Device trace: 136 kernels + 20 copies; all attributed. Each baseline update/replay has eight kernels and one copy; each single-owner path has six kernels and one copy.','',
    'MATCHED LATENCY REDUCTIONS (%)',
    'Candidate / Reference | K | Event median (wins/6) | Wall median | Enqueue median']
for r in groups:lines.append(f"{r['candidate']} / {r['baseline']} | {r['replays_per_cycle']} | {r['event']['median_reduction_percent']:.2f}% ({r['event']['wins']}/6) | {r['wall']['median_reduction_percent']:.2f}% | {r['enqueue']['median_reduction_percent']:.2f}%")
lines+=['Negative reduction means slower. Descriptive matched trial/configuration medians, not confidence intervals.','',
    'FINDINGS',
    'At K=1 versus baseline direct copy: single checked is 42.34% slower, single property-copy is 15.35% slower, and single retained-view copy is 14.37% faster (6/6 wins). At K=2/4/8 retained-view copy reduces event latency by 16.95%/16.52%/17.58%, all 6/6 wins.',
    'The checked API costs time even on the baseline: K=1 event latency is 16.37% higher and host enqueue time is 42.32% higher than baseline direct-copy. Same GPU operations, different host API path.',
    'Removing the checked method while retaining property access reduces K=1 candidate event latency by 17.89%. Retaining the view instead of recreating it reduces that property-copy path by another 27.64%. These are separate ratios with different denominators; do not add them.',
    'The unchanged six-kernel/one-copy candidate traces plus lower host enqueue times support host submission overhead as the reason frequent updates lose. They do not provide an exact additive CPU/GPU breakdown: enqueue overlaps GPU work and CUDA events can include dispatch-induced idle gaps.',
    'The retained-view fast path skips per-call validation and assumes stable owning storage. The test rejects stale handles outside timing. It is not ready for production without a lifecycle contract, so neither draft is promoted.',
    'Static single-owner mapping still wins all 24 event-time comparisons. Frequent updates in Notebooks 06-08 are stress scenarios; they are not established as the actual serving workload.','',
    'SOURCE AUDIT AND NEXT STEP',
    'In the checked-out MuseSpark1xFp4Experts implementation, _expert_map is registered in __init__ and read in _marlin_expert_outputs. Repository text search found no explicit runtime mutation of this specific attribute. This is not proof against indirect buffer manipulation, device moves or future integration.',
    'The next notebook tests static maps with 1/8/32 independent source-prefix stages: eager execution, one graph per stage, and one combined graph containing all stages. Token counts 9/32/128 stay above V4\'s small fused-mapping boundary.',
    'This checks whether prefix savings persist with a different graph boundary and reports complete-chain latency plus normalized per-stage values. The stages are synthetic independent mapping work, not model layers; there is no expert GEMM, communication, attention or artificial filler compute.',
    'No full-model latency, H200 gain, pointer-offset fused-kernel validation, framework-wide buffer compatibility, scheduler initialization or tokenizer initialization is established. Active V4 remains unchanged.','']
(HERE/'NOTEBOOK08_ANALYSIS.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
(HERE/'NOTEBOOK08_REVIEW.txt').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'verified_members':len(hashes),'overall':overall},indent=2))
