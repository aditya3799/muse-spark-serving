import csv,hashlib,io,json,statistics,zipfile
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\OpenKernelChallenge\serving\profiling_kaggle')
archive=next(ROOT.rglob('v4_replay_schedule_20261114T162822Z_f05f16.zip'))
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==len(set(z.namelist()))
    hashes=json.loads(z.read('checksums.json'));assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in hashes.items())
    report=json.loads(z.read('report.json'))
    def table(n):return list(csv.DictReader(io.StringIO(z.read(n).decode())))
    timings,raw,cases,comparisons,trace,devices=map(table,('timings.csv','raw_samples.csv','benchmark_cases.csv','comparisons.csv','trace_variants.csv','trace_device_operations.csv'))
assert report['source_manifest']==json.loads((ROOT/'notebook06_support/source_manifest_06.json').read_text())
assert all(r['status']=='passed' for r in report['sections'].values())
assert (len(timings),len(raw),len(cases),len(comparisons))==(180,1980,30,90)
assert all(c['status']=='passed' for c in cases)
keys=('trial','tokens','dtype','pattern','scope','replays_per_cycle')
samples=defaultdict(list);lookup={}
for r in raw:samples[tuple(r[k] for k in keys)+(r['variant'],)].append(r)
for r in timings:
    key=tuple(r[k] for k in keys)+(r['variant'],);lookup[key]=r;s=samples[key]
    assert len(s)==11 and len({x['sample'] for x in s})==11
    for metric in ('event','wall'):assert abs(statistics.median(float(x[metric+'_us']) for x in s)-float(r[metric+'_us_median']))<1e-8
for r in comparisons:
    key=tuple(r[k] for k in keys);k=int(r['replays_per_cycle'])
    b=lookup[key+('baseline_'+r['scenario'],)];c=lookup[key+('candidate_'+r['scenario'],)]
    for name,row in [('baseline',b),('candidate',c)]:
        assert abs(float(r[name+'_cycle_us'])-float(row['event_us_median']))<1e-8
        assert abs(float(r[name+'_us_per_replay'])-float(row['event_us_median'])/k)<1e-8
    assert abs(float(r['baseline_over_candidate'])-float(b['event_us_median'])/float(c['event_us_median']))<1e-8
assert len(devices)==240 and sum(d['category']=='kernel' for d in devices)==228 and sum(d['category']=='gpu_memcpy' for d in devices)==12
assert all(d['variant']!='unattributed' for d in devices)
assert all(int(r['kernel_count'])==(32 if r['variant']=='mapping/baseline_external' else 25) for r in trace)
assert all(int(r['copy_memset_count'])==(1 if r['variant']=='mapping/baseline_external' else 2) for r in trace)
def summarize(rows):
    vals=[100*(1-1/float(r['baseline_over_candidate'])) for r in rows]
    return {'wins':sum(v>0 for v in vals),'comparisons':len(vals),'median_reduction_percent':statistics.median(vals),'min_reduction_percent':min(vals),'max_reduction_percent':max(vals)}
groups=[]
for scenario in ('static','external','captured'):
    for k in (1,2,4,8,64):groups.append({'scenario':scenario,'replays_per_cycle':k,**summarize([r for r in comparisons if r['scenario']==scenario and r['replays_per_cycle']==str(k)])})
overall={s:summarize([r for r in comparisons if r['scenario']==s]) for s in ('static','external','captured')}
analysis={'run_id':report['run_id'],'zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'member_checksums_verified':len(hashes),
    'event_and_wall_rows_recomputed':180,'comparison_rows_verified':90,'groups':groups,'overall':overall,
    'decision':'Two-buffer cache is useful for static/long-reuse maps, but external eager refresh has a serious frequent-update penalty. Test single-owner aliasing to remove explicit refresh; do not promote current draft universally.',
    'scope':'Source-sliced prefix graphs and synthetic GPU-resident map updates. Production frequency, full model and H200 behavior unmeasured.'}
lines=['NOTEBOOK 06: VERIFIED REPLAY-SCHEDULE REVIEW','Run: '+report['run_id'],'',
    'VALIDATION','All nine file checksums and source provenance matched. Recomputed 180 event/wall medians from 1,980 samples; verified 90 comparisons. All 192 inference-mode input cases, 30 benchmark cases and 360 changed-input schedule checks passed.',
    'GPU trace: 228 kernels + 12 copies, all assigned. Annotation spans excluded.','',
    'SCHEDULE RESULTS: median within-trial latency reduction (%)',
    'Scenario | Replays/update-cycle | Wins/6 | Median | Range']
for r in groups:lines.append(f"{r['scenario']} | {r['replays_per_cycle']} | {r['wins']}/{r['comparisons']} | {r['median_reduction_percent']:.2f}% | {r['min_reduction_percent']:.2f}% to {r['max_reduction_percent']:.2f}%")
lines+=['Negative reductions mean slower candidate latency. Static has no update; captured combines update plus K prefixes in a single graph; external updates eagerly then launches K existing graphs.','',
    'FINDINGS',
    f"Static maps: {overall['static']['wins']}/30 wins, median reduction {overall['static']['median_reduction_percent']:.2f}%. The cached mapping itself continues to help.",
    'External refresh: K=1 was 141.63% slower at the median (roughly 2.4x baseline latency); K=2 was 102.27% slower; K=4 was 34.02% slower. K=8 was effectively mixed (3/6 wins, median +0.15% reduction). K=64 won 6/6 with a 15.06% median reduction.',
    'These are descriptive matched configuration/trial medians, not confidence intervals or a universal crossover. Incoming maps already resided on GPU; no CPU data preparation or host-to-device input transfer was timed.',
    'Captured update cycles won all 30 comparisons, but remove host update dispatch from replay. They cannot be substituted for the external-update schedule when assessing real refresh overhead.',
    'At K=4 the external trace showed baseline 32 kernels + one copy per cycle, versus candidate 25 kernels + two copies. The candidate saves mapping kernels but adds a fill/copy refresh path and host API work. Trace durations cannot be subtracted from benchmark timings to assign exact CPU/GPU percentages.',
    'Clock/host variability remains visible across trials; avoid comparing absolute times between notebooks.','',
    'DECISION AND NEXT EXPERIMENT',
    'Keep the two-buffer draft restricted to static or sufficiently reused maps pending integration evidence. Do not refresh it on every forward or promote it as universally faster.',
    'Notebook 07 introduces a second, unapplied draft with a single owning lookup buffer. _expert_map becomes a property view into lookup[1:]. The same forward arithmetic is retained; updates copy into the view once, with no separate refresh. Direct in-place map writes should also update lookup contents immediately.',
    'Compare baseline, prior two-buffer cache and new single-owner cache within the same run, under static and external-update schedules at K=1/2/4/8/64. Verify aliasing, device moves, direct-write graph visibility, rejected attribute replacement, checkpoint exclusion and ownership bytes.',
    'Compatibility limits: _expert_map is no longer a registered buffer, so framework-wide introspection needs review. Its data pointer starts four bytes into the owning lookup; actual fused-kernel correctness/performance must still be validated on supported hardware. This notebook checks only source slices and branch metadata.',
    'Active V4 and the earlier draft remain unchanged. No full-serving/H200 gain, real alignment/GEMM validation or startup profile has been established.','']
(HERE/'NOTEBOOK06_ANALYSIS.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
(HERE/'NOTEBOOK06_REVIEW.txt').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'verified':True,'overall':overall,'external':[r for r in groups if r['scenario']=='external']},indent=2))
