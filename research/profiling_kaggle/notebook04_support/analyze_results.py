import csv,hashlib,io,json,statistics,zipfile
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\OpenKernelChallenge\serving\profiling_kaggle')
archive=next(ROOT.rglob('v4_mapping_integration_20261114T152922Z_58a8e0.zip'))
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==len(set(z.namelist()))
    hashes=json.loads(z.read('checksums.json'))
    assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in hashes.items())
    report=json.loads(z.read('report.json'))
    def table(n):return list(csv.DictReader(io.StringIO(z.read(n).decode())))
    timings,raw,cases,comparisons,trace,device=map(table,('timings.csv','raw_samples.csv','benchmark_cases.csv','comparisons.csv','trace_variants.csv','trace_device_operations.csv'))
assert report['source_manifest']==json.loads((ROOT/'notebook04_support/source_manifest_04.json').read_text())
assert all(r['status']=='passed' for r in report['sections'].values())
assert all(r['status']=='passed' for r in report['sections']['lifecycle']['checks'])
assert (len(timings),len(raw),len(cases),len(comparisons))==(204,2244,48,96)
keys=('trial','tokens','dtype','pattern','scope')
samples=defaultdict(list);lookup={}
for r in raw:samples[tuple(r[k] for k in keys)+(r['variant'],)].append(r)
for r in timings:
    key=tuple(r[k] for k in keys)+(r['variant'],);s=samples[key];lookup[key]=r
    assert len(s)==11 and len({x['sample'] for x in s})==11
    for metric in ('event','wall'):assert abs(statistics.median(float(x[metric+'_us']) for x in s)-float(r[metric+'_us_median']))<1e-8
for r in comparisons:
    key=tuple(r[k] for k in keys)
    b=lookup[key+('baseline_module_'+r['execution'],)];c=lookup[key+('cached_module_'+r['execution'],)]
    assert abs(float(r['baseline_us'])-float(b['event_us_median']))<1e-8
    assert abs(float(r['candidate_us'])-float(c['event_us_median']))<1e-8
    assert abs(float(r['baseline_over_candidate'])-float(r['baseline_us'])/float(r['candidate_us']))<1e-8
assert len(device)==72 and all(r['category']=='kernel' and r['variant']!='unattributed' for r in device)
assert all(int(r['kernel_count'])==(7 if r['variant']=='mapping/baseline_module' else 5) for r in trace)
summary={}
for mode in ('eager','graph'):
    rows=[r for r in comparisons if r['execution']==mode];red=[100*(1-1/float(r['baseline_over_candidate'])) for r in rows]
    saved=[float(r['baseline_us'])-float(r['candidate_us']) for r in rows]
    summary[mode]={'wins':sum(v>0 for v in red),'comparisons':len(rows),'median_reduction_percent':statistics.median(red),'min_reduction_percent':min(red),'max_reduction_percent':max(red),'median_saved_us':statistics.median(saved)}
focused=[]
for n in ('9','32','128','256'):
    rows=[r for r in comparisons if (r['tokens'],r['dtype'],r['pattern'],r['execution'])==(n,'int64','hot_local','graph')]
    focused.append({'tokens':int(n),'baseline_us':statistics.median(float(r['baseline_us']) for r in rows),'candidate_us':statistics.median(float(r['candidate_us']) for r in rows),'median_within_trial_reduction_percent':statistics.median(100*(1-1/float(r['baseline_over_candidate'])) for r in rows)})
updates={name:statistics.median(float(r['event_us_median']) for r in timings if r['variant']==name) for name in ('baseline_update_eager','cached_update_eager','baseline_update_graph','cached_update_graph')}
losses=[]
for r in comparisons:
    if float(r['baseline_over_candidate'])>=1:continue
    key=tuple(r[k] for k in keys)
    b={s['sample']:float(s['event_us']) for s in samples[key+('baseline_module_'+r['execution'],)]}
    c={s['sample']:float(s['event_us']) for s in samples[key+('cached_module_'+r['execution'],)]}
    losses.append(dict(r,paired_sample_wins=sum(b[k]>c[k] for k in b),paired_samples=len(b)))
analysis={'run_id':report['run_id'],'zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'checksums_verified':len(hashes),
          'timing_rows_recomputed':204,'comparison_rows_verified':96,'summary':summary,'focused_graph':focused,'update_us':updates,'losses':losses,
          'decision':'Module cache remains promising under graph replay, with an explicit refresh contract. Next test real source slices from an unapplied draft and direct update-plus-reuse cycles. Do not infer exact break-even by adding isolated medians.'}
lines=['NOTEBOOK 04: VERIFIED RESULTS REVIEW','Run: '+report['run_id'],'',
    'All 11 artifact checksums and the source manifest matched. Recomputed all 204 event and wall medians from 2,244 samples and verified all 96 comparison rows.',
    'All nine lifecycle groups passed, including 672 input configurations / 1,344 module outputs, CPU/GPU moves, nonpersistent-buffer checkpoint semantics, explicit refresh, graph updates, and producer/consumer stream ordering.',
    'All 48 benchmark cases and 288 benchmark changed-input graph checks passed.','',
    'FORWARD RESULTS (matched configuration/trial medians)']
for mode,r in summary.items():lines.append(f"{mode}: {r['wins']}/{r['comparisons']} wins; median reduction {r['median_reduction_percent']:.2f}%; range {r['min_reduction_percent']:.2f}% to {r['max_reduction_percent']:.2f}%; median saving {r['median_saved_us']:.2f} us.")
lines+=['','INT64 / HOT-LOCAL / GRAPH: median of three trial medians','Tokens | Baseline us | Cached us']
for r in focused:lines.append(f"{r['tokens']} | {r['baseline_us']:.2f} | {r['candidate_us']:.2f}")
lines+=['','EXCEPTION RETAINED',
    'Trial 1, 128 tokens/int64/hot-local eager: baseline median 119.71 us, candidate 128.30 us (7.18% slower). Both series had substantial temporal variation; the candidate won 10 of the 11 matched sample pairs. This is not silently discarded or relabeled a win. Graph comparisons all favored the candidate.',
    'Three trials share one Kaggle runtime, not independent hardware sessions.','',
    'UPDATE COSTS (median of trial medians, CUDA-event intervals)']
for name,value in updates.items():lines.append(f'{name}: {value:.2f} us')
lines+=['The candidate update API copies the map and refreshes the lookup; baseline only copies the map. The extra eager update cost is comparable to a single forward saving. Notebook 05 therefore measures an update plus 1/8/64 mapping prefixes together instead of adding separately measured medians.',
    'Trace: 72 real kernels total. Each baseline prefix used seven kernels versus five for cached lookup. Annotation spans excluded; no serving-percentage inference.','',
    'LIFECYCLE CONTRACT',
    'Direct writes to _expert_map still produce stale cached results until refresh. The test intentionally demonstrated this failure mode before checking the remedy. Updates must use the update API or explicit refresh; consumers need stream ordering. Buffer replacement/device movement requires graph recapture.',
    'Checkpoint testing covered a small marker and nonpersistent mapping buffers, not a model checkpoint or meta-device load.','',
    'NEXT EXPERIMENT: NOTEBOOK 05',
    'Prepared a V4-relative cached-mapping patch draft without applying it. Static checks require the fused <=64-route guard, alignment/GEMM tail and unrelated methods to remain unchanged.',
    'Notebook 05 executes constructor and pre-alignment source slices from baseline and draft, tests branch selection and exact fallback values, then benchmarks unchanged-map calls and update/reuse cycles.',
    'For fused small-route cases, the source merely allocates a buffer that the omitted kernel would populate. Those cases check branch/shape/dtype only and are not timed.',
    'This is progress toward integration, not evidence that the full serving class, alignment, GEMMs or H200 run passed. Active V4 source remains unchanged.','']
(HERE/'NOTEBOOK04_ANALYSIS.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
(HERE/'NOTEBOOK04_REVIEW.txt').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'verified':True,'summary':summary,'updates':updates},indent=2))
