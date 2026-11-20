import csv,hashlib,io,json,statistics,zipfile
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\OpenKernelChallenge\serving\profiling_kaggle')
archive=next(ROOT.rglob('v4_graph_boundaries_20261114T180926Z_ac3144.zip'))
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==len(set(z.namelist()))
    hashes=json.loads(z.read('checksums.json'));assert set(hashes)==set(z.namelist())-{'checksums.json'}
    for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    report=json.loads(z.read('report.json'))
    def table(n):return list(csv.DictReader(io.StringIO(z.read(n).decode())))
    timings,raw,cases,comparisons,trace,devices=map(table,('timings.csv','raw_samples.csv','benchmark_cases.csv','comparisons.csv','trace_variants.csv','trace_device_operations.csv'))
assert report['source_manifest']==json.loads((ROOT/'notebook09_support/source_manifest_09.json').read_text())
assert all(r['status']=='passed' for r in report['sections'].values())
assert (len(timings),len(raw),len(cases),len(comparisons))==(162,1782,27,135)
assert all(c['status']=='passed' for c in cases)
assert sum(int(c['changed_input_output_checks']) for c in cases)==4428
keys=('trial','tokens','dtype','pattern','scope','stages');samples=defaultdict(list);lookup={}
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
        for label,value in [('baseline',b),('candidate',c)]:
            assert abs(float(r[metric+'_'+label+'_us'])-value)<1e-8
            assert abs(float(r[metric+'_'+label+'_us_per_stage'])-value/depth)<1e-8
        assert abs(float(r[metric+'_reduction_percent'])-100*(1-c/b))<1e-8
assert len(devices)==1008 and all(d['category']=='kernel' and d['variant']!='unattributed' for d in devices)
assert len(trace)==18
for r in trace:assert (int(r['kernel_count']),int(r['copy_memset_count']))==(64 if r['variant'].startswith('mapping/baseline') else 48,0)
def summarize(rows,metric):
    vals=[float(r[metric+'_reduction_percent']) for r in rows]
    return {'wins':sum(v>0 for v in vals),'comparisons':len(vals),'median_reduction_percent':statistics.median(vals),'min_reduction_percent':min(vals),'max_reduction_percent':max(vals)}
groups=[];overall=[]
for candidate,baseline in sorted(set((r['candidate'],r['baseline']) for r in comparisons)):
    selected=[r for r in comparisons if (r['candidate'],r['baseline'])==(candidate,baseline)]
    overall.append(dict(candidate=candidate,baseline=baseline,**{m:summarize(selected,m) for m in ('event','wall','enqueue')}))
    for d in (1,8,32):groups.append(dict(candidate=candidate,baseline=baseline,stages=d,**{m:summarize([r for r in selected if int(r['stages'])==d],m) for m in ('event','wall','enqueue')}))
analysis={'run_id':report['run_id'],'zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'member_checksums_verified':len(hashes),'timing_rows_recomputed_all_three_metrics':162,'comparison_rows_verified':135,'groups':groups,'overall':overall,
    'decision':'Mapping savings persist inside combined graphs, but benchmark inputs add dtype conversions absent from the router default output contract. Next test the four ID/score dtype combinations within matched runs before any integration claim.'}
lines=['NOTEBOOK 09: VERIFIED GRAPH-BOUNDARY REVIEW','Run: '+report['run_id'],'',
    'VALIDATION',f'All {len(hashes)} ZIP member checksums and source provenance match. Recomputed all three metrics for 162 rows from 1,782 samples and verified 135 comparisons. All 27 cases, 4,428 changed-input output checks and inherited 576 output checks passed.',
    'Trace: 1,008 kernels, no copies, no unattributed device operations. Each eight-stage baseline cycle has 64 kernels, each single-owner cycle 48, under all three schedules.','',
    'LATENCY REDUCTION (%)',
    'Candidate / Reference | Stages | Event median (wins/9) | Wall median | Enqueue median']
for r in groups:lines.append(f"{r['candidate']} / {r['baseline']} | {r['stages']} | {r['event']['median_reduction_percent']:.2f}% ({r['event']['wins']}/9) | {r['wall']['median_reduction_percent']:.2f}% | {r['enqueue']['median_reduction_percent']:.2f}%")
lines+=['Negative reductions mean slower; descriptive matched trial/configuration ratios, not confidence intervals.','',
    'FINDINGS',
    'Inside one combined graph, the single-owner mapping wins all 27 comparisons: 18.30% median reduction at one stage, 19.28% at eight, 19.21% at 32. Thus the benefit is not solely from replaying separate graphs.',
    'Combining 32 graphs improves baseline event latency by 6.63% (8/9 wins), and candidate latency by 8.05% (9/9 wins). Their host enqueue medians fall by 63.65% and 68.98%; those percentages are not whole-cycle speedups.',
    'The one-stage combined-versus-separate comparison is a same-work control: both replay one graph. Small differences there demonstrate timing/allocation variability rather than graph coalescing.',
    'The operation-count reduction remains two kernels per prefix. The trace counts 64 versus 48 kernels across eight prefixes with zero refresh copies.','',
    'INPUT-CONTRACT LIMITATION FOUND IN SOURCE AUDIT',
    'Notebook 09 uses int64 IDs and float16 scores inherited from the earlier harness. The inspected JIT and Triton router allocation paths default to contiguous int32 IDs and float32 scores. The serving prefix converts IDs to int32 and scores to float32, so the old harness includes conversion work that is normally unnecessary for those default outputs.',
    'This does not invalidate matched synthetic comparisons. It limits how representative their absolute times, kernel counts and percentage gains are. The actual backend, preallocated output buffers and runtime layout have not been observed in a full model run.',
    'Notebook 10 will test all four ID/score dtype combinations in the same run: int64/fp16 historical control, int32/fp16 and int64/fp32 individual conversion controls, and int32/fp32 matching the inspected default router outputs.',
    'Only baseline and unchanged single-owner prefixes are timed, inside combined graphs, with 1/8 stages and 9/32/128 tokens. The actual router kernel remains omitted.','',
    'BOUNDARIES',
    'All stages are independent mapping prefixes, not transformer layers. No alignment, expert GEMM, attention, communication or full model is executed. Small fused mapping, framework buffer compatibility, H200 performance and scheduler/tokenizer startup remain unvalidated. Active V4 and both drafts stay unchanged.','']
(HERE/'NOTEBOOK09_ANALYSIS.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
(HERE/'NOTEBOOK09_REVIEW.txt').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'verified_members':len(hashes),'overall':overall},indent=2))
