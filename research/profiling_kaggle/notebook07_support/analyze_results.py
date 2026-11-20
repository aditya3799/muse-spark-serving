import csv,hashlib,io,json,statistics,zipfile
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\OpenKernelChallenge\serving\profiling_kaggle')
archive=next(ROOT.rglob('v4_single_owner_20261114T173941Z_88f16c.zip'))
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==len(set(z.namelist()))
    hashes=json.loads(z.read('checksums.json'))
    assert set(hashes)==set(z.namelist())-{'checksums.json'}
    for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    report=json.loads(z.read('report.json'))
    def table(n):return list(csv.DictReader(io.StringIO(z.read(n).decode())))
    timings,raw,cases,comparisons,trace,devices=map(table,('timings.csv','raw_samples.csv','benchmark_cases.csv','comparisons.csv','trace_variants.csv','trace_device_operations.csv'))
assert report['source_manifest']==json.loads((ROOT/'notebook07_support/source_manifest_07.json').read_text())
assert all(r['status']=='passed' for r in report['sections'].values())
assert (len(timings),len(raw),len(cases),len(comparisons))==(180,1980,30,120)
assert all(c['status']=='passed' for c in cases)
assert sum(int(c['changed_input_schedule_checks']) for c in cases)==360
assert all(c['status']=='passed' for c in report['sections']['lifecycle']['checks'])
keys=('trial','tokens','dtype','pattern','scope','replays_per_cycle')
samples=defaultdict(list);lookup={}
for r in raw:samples[tuple(r[k] for k in keys)+(r['variant'],)].append(r)
for r in timings:
    key=tuple(r[k] for k in keys)+(r['variant'],);assert key not in lookup;lookup[key]=r;s=samples[key]
    assert len(s)==11 and len({x['sample'] for x in s})==11
    for metric in ('event','wall'):assert abs(statistics.median(float(x[metric+'_us']) for x in s)-float(r[metric+'_us_median']))<1e-8
for r in comparisons:
    key=tuple(r[k] for k in keys);k=int(r['replays_per_cycle'])
    b=lookup[key+('baseline_'+r['scenario'],)];c=lookup[key+(r['candidate']+'_'+r['scenario'],)]
    for name,row in [('baseline',b),('candidate',c)]:
        assert abs(float(r[name+'_cycle_us'])-float(row['event_us_median']))<1e-8
        assert abs(float(r[name+'_us_per_replay'])-float(row['event_us_median'])/k)<1e-8
    assert abs(float(r['baseline_over_candidate'])-float(b['event_us_median'])/float(c['event_us_median']))<1e-8
    assert abs(float(r['wall_saved_us'])-(float(b['wall_us_median'])-float(c['wall_us_median'])))<1e-8
assert len(devices)==340 and sum(d['category']=='kernel' for d in devices)==324 and sum(d['category']=='gpu_memcpy' for d in devices)==16
assert all(d['variant']!='unattributed' for d in devices)
expected={'mapping/baseline_external':(32,1),'mapping/two_buffer_external':(25,2),'mapping/single_owner_external':(24,1)}
assert len(trace)==12
for r in trace:assert (int(r['kernel_count']),int(r['copy_memset_count']))==expected[r['variant']]
def summarize(vals):
    return {'wins':sum(v>0 for v in vals),'comparisons':len(vals),'median_reduction_percent':statistics.median(vals),'min_reduction_percent':min(vals),'max_reduction_percent':max(vals)}
def reductions(rows):return [100*(1-1/float(r['baseline_over_candidate'])) for r in rows]
groups=[];overall={}
for candidate in ('two_buffer','single_owner'):
    for scenario in ('static','external'):
        selected=[r for r in comparisons if r['candidate']==candidate and r['scenario']==scenario]
        overall[candidate+'_'+scenario]=summarize(reductions(selected))
        for k in (1,2,4,8,64):groups.append(dict(candidate=candidate,scenario=scenario,replays_per_cycle=k,**summarize(reductions([r for r in selected if int(r['replays_per_cycle'])==k]))))
direct=[]
for case in cases:
    key=tuple(case[k] for k in keys)
    for scenario in ('static','external'):
        old=float(lookup[key+('two_buffer_'+scenario,)]['event_us_median'])
        new=float(lookup[key+('single_owner_'+scenario,)]['event_us_median'])
        direct.append(dict(scenario=scenario,reduction_percent=100*(1-new/old)))
versus_prior={s:summarize([r['reduction_percent'] for r in direct if r['scenario']==s]) for s in ('static','external')}
analysis={'run_id':report['run_id'],'zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'member_checksums_verified':len(hashes),'event_and_wall_rows_recomputed':180,'comparison_rows_verified':120,'groups':groups,'overall':overall,'single_owner_vs_two_buffer':versus_prior,'trace_per_cycle':expected,'decision':'Single ownership removes the extra fill/copy and improves external-update results, but does not eliminate frequent-update latency regressions. Next isolate host update API/property costs with matched baseline controls; no source promotion.'}
lines=['NOTEBOOK 07: VERIFIED SINGLE-OWNER REVIEW','Run: '+report['run_id'],'',
    'VALIDATION',f'All {len(hashes)} member checksums and source provenance matched. Recomputed 180 timing rows from 1,980 raw samples; verified 120 comparisons. All 576 output checks across 192 configurations, 30 benchmark cases and 360 schedule checks passed.',
    'Aliasing, CPU/GPU moves, direct mutation, graph visibility, rejected attribute replacement and checkpoint exclusion checks passed. Map offset remains four bytes; real fused-kernel compatibility is NOT established.',
    'Trace: 324 kernels and 16 copies; no unattributed device operations. Annotation spans excluded.','',
    'LATENCY REDUCTION AGAINST ORIGINAL BASELINE (%)',
    'Candidate | Schedule | Replays per update cycle | Wins | Median | Range']
for r in groups:lines.append(f"{r['candidate']} | {r['scenario']} | {r['replays_per_cycle']} | {r['wins']}/{r['comparisons']} | {r['median_reduction_percent']:.2f}% | {r['min_reduction_percent']:.2f}% to {r['max_reduction_percent']:.2f}%")
lines+=['Negative reduction means higher/slower latency. Static schedules have no updates. Values are medians of matched configuration/trial ratios, not confidence intervals or universal crossover points.','',
    'INTERPRETATION',
    f"Single-owner static: {overall['single_owner_static']['wins']}/30 wins, median latency reduction {overall['single_owner_static']['median_reduction_percent']:.2f}%.",
    'Single-owner external: K=1 remains 46.12% slower; K=2 remains 15.00% slower. K=4 wins 5/6 (median 4.77% lower latency); K=8 and K=64 win 6/6 (11.01% and 16.63% lower latency).',
    f"Versus the prior two-buffer candidate, the single-owner external schedule wins {versus_prior['external']['wins']}/30 comparisons, median reduction {versus_prior['external']['median_reduction_percent']:.2f}%.",
    'At K=4, per trace cycle: baseline 32 kernels + 1 copy, two-buffer 25 kernels + 2 copies, single-owner 24 kernels + 1 copy. This supports removal of the extra fill/copy. It does not assign exact host/GPU percentages to benchmark latency.',
    'Residual overhead could include checked update method dispatch, the no_grad wrapper, validation and creating the map view. This is a hypothesis for Notebook 08, not a measured decomposition.',
    'Baseline external update uses a direct copy whereas the candidate uses a checked method. Notebook 08 adds the same checked method to the baseline as an experimental control and measures direct-property and retained-view writes separately.',
    'Schedule checks in Notebook 07 preload each module with the expected map before calling the update variant. They verify output consistency but do not independently prove each update wrapper changed old state. Notebook 08 explicitly poisons the old map before each update check; separate Notebook 07 direct-write tests already demonstrate shared-storage visibility.','',
    'NEXT EXPERIMENT: 08_kaggle_update_overhead.ipynb',
    'Keep the Notebook 07 source slices unchanged. Compare baseline direct/checked updates and single-owner checked/property-copy/retained-view-copy updates, plus two static graph controls. Test K=1/2/4/8 at 32/128 tokens in three shuffled trials.',
    'Report CUDA-event cycle time, synchronized wall time and host enqueue time separately. Host enqueue overlaps device execution and is not an additive component to subtract from event timing.',
    'Retained-view writes are diagnostic: bind only after device placement, validate the input contract before use and rebind/recapture after storage changes. Do not bypass validation in production based on this result.',
    'All variants remain unapplied. T4 source-prefix results do not establish full-serving/H200 speedup, actual alignment/GEMM behavior, scheduler/tokenizer startup or framework-wide buffer compatibility.','']
(HERE/'NOTEBOOK07_ANALYSIS.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
(HERE/'NOTEBOOK07_REVIEW.txt').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'verified_members':len(hashes),'overall':overall,'versus_prior':versus_prior},indent=2))
