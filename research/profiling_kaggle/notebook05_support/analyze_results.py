import csv,hashlib,io,json,statistics,zipfile
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\OpenKernelChallenge\serving\profiling_kaggle')
archive=next(ROOT.rglob('v4_source_candidate_20261114T160139Z_2d5e90.zip'))
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==len(set(z.namelist()))
    hashes=json.loads(z.read('checksums.json'))
    assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in hashes.items())
    report=json.loads(z.read('report.json'))
    def table(n):return list(csv.DictReader(io.StringIO(z.read(n).decode())))
    timings,raw,cases,comparisons,trace,devices=map(table,('timings.csv','raw_samples.csv','benchmark_cases.csv','comparisons.csv','trace_variants.csv','trace_device_operations.csv'))
assert report['source_manifest']==json.loads((ROOT/'notebook05_support/source_manifest_05.json').read_text())
assert all(s['status']=='passed' for s in report['sections'].values())
assert (len(timings),len(raw),len(cases),len(comparisons))==(264,2904,66,132)
assert all(c['status']=='passed' for c in cases)
keys=('trial','tokens','dtype','pattern','scope','calls_per_update')
samples=defaultdict(list);lookup={}
for r in raw:samples[tuple(r[k] for k in keys)+(r['variant'],)].append(r)
for r in timings:
    key=tuple(r[k] for k in keys)+(r['variant'],);lookup[key]=r;s=samples[key]
    assert len(s)==11 and len({x['sample'] for x in s})==11
    for metric in ('event','wall'):assert abs(statistics.median(float(x[metric+'_us']) for x in s)-float(r[metric+'_us_median']))<1e-8
for r in comparisons:
    key=tuple(r[k] for k in keys);k=int(r['calls_per_update']) or 1
    b=lookup[key+('baseline_'+r['execution'],)];c=lookup[key+('candidate_'+r['execution'],)]
    for name,row in [('baseline',b),('candidate',c)]:
        assert abs(float(r[name+'_event_us'])-float(row['event_us_median']))<1e-8
        assert abs(float(r[name+'_us_per_mapping_including_update'])-float(row['event_us_median'])/k)<1e-8
    assert abs(float(r['baseline_over_candidate'])-float(b['event_us_median'])/float(c['event_us_median']))<1e-8
assert len(devices)==84 and all(d['category']=='kernel' and d['variant']!='unattributed' for d in devices)
assert all(int(r['kernel_count'])==(8 if r['variant']=='mapping/baseline_source' else 6) for r in trace)
def summarize(rows):
    vals=[100*(1-1/float(r['baseline_over_candidate'])) for r in rows]
    return {'wins':sum(x>0 for x in vals),'comparisons':len(vals),'reduction_percent_median':statistics.median(vals),'reduction_percent_min':min(vals),'reduction_percent_max':max(vals)}
prefix={mode:summarize([r for r in comparisons if r['scope']=='source_prefix' and r['execution']==mode]) for mode in ('eager','graph')}
cadence=[]
for k in (1,8,64):
    for mode in ('eager','graph'):
        cadence.append({'calls_per_update':k,'execution':mode,**summarize([r for r in comparisons if (r['scope'],r['calls_per_update'],r['execution'])==('update_cadence',str(k),mode)])})
losses=[r for r in comparisons if float(r['baseline_over_candidate'])<1]
analysis={'run_id':report['run_id'],'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'member_checksums_verified':len(hashes),
    'event_and_wall_median_rows_recomputed':264,'comparison_rows_verified':132,'prefix':prefix,'cadence':cadence,'losses_retained':losses,
    'decision':'Retain candidate for reused/static maps; do not refresh on every eager forward. Next gate measures eager refresh outside existing graphs, under inference_mode, with K=1/2/4/8/64 replays.',
    'limits':'Source slices only. Internal-update cadence graphs combine the update and K prefixes in one capture. They do not predict separately launched refresh plus existing graph replay or H200/model latency.'}
lines=['NOTEBOOK 05: VERIFIED RESULTS REVIEW','Run: '+report['run_id'],'',
    'VALIDATION','All nine artifact checksums and the source manifest matched. Recomputed 264 event/wall median pairs from 2,904 samples; verified 132 comparison rows and normalized cycle costs.',
    'All 66 benchmark/cadence cases passed: 288 prefix changed-input graph checks and 72 cadence checks. Source checks included 576 actual fallback cases and 96 deferred-fused cases. Deferred cases checked branch/metadata/scores only; no alignment kernel ran.',
    'Trace contained 84 actual kernels, all attributed; eight per baseline prefix versus six per candidate prefix. No annotation spans were counted as kernels.','',
    'UNCHANGED-MAP SOURCE PREFIX']
for mode,r in prefix.items():lines.append(f"{mode}: {r['wins']}/{r['comparisons']} wins; median latency reduction {r['reduction_percent_median']:.2f}% (range {r['reduction_percent_min']:.2f}% to {r['reduction_percent_max']:.2f}%).")
lines+=['','UPDATE PLUS K PREFIX CALLS: directly measured cycles','K | execution | wins/6 | median latency reduction | range']
for r in cadence:lines.append(f"{r['calls_per_update']} | {r['execution']} | {r['wins']}/{r['comparisons']} | {r['reduction_percent_median']:.2f}% | {r['reduction_percent_min']:.2f}% to {r['reduction_percent_max']:.2f}%")
lines+=['','INTERPRETATION',
    'At K=1, every eager comparison regressed: median slowdown 13.29%, range 11.44%-20.77%. Refreshing every forward erases the cached-path benefit in this workload.',
    'At K=8 and K=64, all eager and captured-cycle graph comparisons favored the candidate. Do not infer that K=8 is the exact crossover: K=2 and K=4 were not measured here.',
    'K=1 captured-cycle graph results were marginal: five of six wins; one 128-token trial regressed by 0.59%. The entire update and prefix sequence was captured in a single graph, which removes host-side update dispatch from replay.',
    'The existing V4 class constructs _expert_map during initialization; the inspected file showed no later writes. Static reuse is plausible, but runtime mapping update frequency and all external writers have not been established by a model run.',
    'Three trials occurred within one Kaggle session. Absolute graph times drifted substantially across some trials; use matching within-trial comparisons, not pooled absolute timings or previous-notebook ratios.','',
    'NEXT EXPERIMENT: NOTEBOOK 06',
    'Keep the same unapplied source candidate. Compare three schedules under torch.inference_mode: static map + K separate graph replays; eager update outside capture + K separate replays; and one captured update-plus-K cycle as a control.',
    'Test K=1,2,4,8,64 at 32/128 tokens, with interleaved baseline/candidate variants and three shuffled trials. Primary gate: external update scenario, where refresh overhead is paid before replaying an already-captured graph.',
    'Correctness covers buffers created normally and within inference mode, with updates/forwards in inference mode. Incoming maps already reside on GPU; CPU data preparation is not included.',
    'This is a schedule test on source-sliced prefixes, not a full-model graph. It supplies no H200 speedup, serving fraction, actual alignment/GEMM validation, or startup profile.',
    'Active V4 and the unapplied patch remain unchanged. Do not promote or submit the candidate automatically.','']
(HERE/'NOTEBOOK05_ANALYSIS.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
(HERE/'NOTEBOOK05_REVIEW.txt').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'verified':True,'prefix':prefix,'cadence':cadence},indent=2))
