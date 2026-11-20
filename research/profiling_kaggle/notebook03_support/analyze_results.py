"""Verify notebook-03 artifacts and summarize matched within-run comparisons."""
import csv
import hashlib
import io
import json
import statistics
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

archive, root, out = map(Path, sys.argv[1:4])
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist()) == len(set(z.namelist()))
    hashes = json.loads(z.read('checksums.json'))
    verified = {n: hashlib.sha256(z.read(n)).hexdigest() == h for n,h in hashes.items()}
    assert all(verified.values())
    report = json.loads(z.read('report.json'))
    def table(name):
        return list(csv.DictReader(io.StringIO(z.read(name).decode())))
    timings, raw, cases, comparisons, trace, devices = map(table, ('timings.csv','raw_samples.csv','benchmark_cases.csv','comparisons.csv','trace_variants.csv','trace_device_operations.csv'))
assert report['source_manifest'] == json.loads((root/'notebook03_support/source_manifest_03.json').read_text())
assert all(s['status']=='passed' for s in report['sections'].values())
assert (len(timings),len(raw),len(cases),len(comparisons)) == (588,6468,96,384)
assert all(c['status']=='passed' for c in cases)
assert sum(int(c['changed_input_graph_checks']) for c in cases)==1152
assert report['sections']['correctness']['input_cases']==672
assert report['sections']['setup_cost']['changed_mapping_graph_checks']==6
keys = ('trial','tokens','dtype','pattern','scope')
groups = defaultdict(list)
for r in raw:
    groups[tuple(r[k] for k in keys)+ (r['variant'],)].append(r)
by_key = {}
for r in timings:
    key = tuple(r[k] for k in keys)+(r['variant'],)
    assert key not in by_key
    by_key[key]=r
    samples=groups[key]
    assert len(samples)==11 and len({s['sample'] for s in samples})==11
    for metric in ('event','wall'):
        assert abs(statistics.median(float(s[metric+'_us']) for s in samples)-float(r[metric+'_us_median']))<1e-8
for r in comparisons:
    key = tuple(r[k] for k in keys)
    baseline = by_key[key+('baseline_'+r['execution'],)]
    candidate = by_key[key+(r['candidate']+'_'+r['execution'],)]
    b,c=float(baseline['event_us_median']),float(candidate['event_us_median'])
    assert abs(float(r['baseline_event_us'])-b)<1e-8
    assert abs(float(r['candidate_event_us'])-c)<1e-8
    assert abs(float(r['baseline_over_candidate'])-b/c)<1e-8
assert len(devices)==114 and all(r['category']=='kernel' and r['variant']!='unattributed' for r in devices)
assert all(int(r['kernel_count'])==(5 if r['variant']=='mapping/cached_lookup' else 7) for r in trace)

def distribution(values):
    return {'min':min(values),'median':statistics.median(values),'max':max(values)}

summaries=[]
for scope in ('helper_only','fallback_cast_map'):
    for candidate in ('cached_lookup','rebuilt_lookup'):
        for mode in ('eager','graph'):
            rows=[r for r in comparisons if (r['scope'],r['candidate'],r['execution'])==(scope,candidate,mode)]
            summaries.append({'scope':scope,'candidate':candidate,'execution':mode,'comparisons':len(rows),
                'event_median_wins':sum(float(r['baseline_over_candidate'])>1 for r in rows),
                'wall_median_wins':sum(float(r['wall_saved_us'])>0 for r in rows),
                'latency_reduction_percent':distribution([(1-1/float(r['baseline_over_candidate']))*100 for r in rows]),
                'saved_event_us':distribution([float(r['event_saved_us']) for r in rows])})
focused=[]
for tokens in ('9','32','128','256'):
    rows=sorted([r for r in comparisons if (r['tokens'],r['dtype'],r['scope'],r['pattern'],r['candidate'],r['execution'])==(tokens,'int64','fallback_cast_map','hot_local','cached_lookup','graph')],key=lambda r:int(r['trial']))
    reductions=[(1-1/float(r['baseline_over_candidate']))*100 for r in rows]
    focused.append({'tokens':int(tokens),'baseline_us_median_of_trial_medians':statistics.median(float(r['baseline_event_us']) for r in rows),
        'candidate_us_median_of_trial_medians':statistics.median(float(r['candidate_event_us']) for r in rows),
        'within_trial_reduction_percent':reductions,'median_within_trial_reduction_percent':statistics.median(reductions)})
setup={variant:statistics.median(float(r['event_us_median']) for r in timings if r['variant']==variant and r['scope']=='lookup_setup')
       for variant in ('build_table_eager','build_table_graph','refresh_existing_table_eager','refresh_existing_table_graph')}
trace_summary={name:statistics.median(float(r['kernel_sum_us']) for r in trace if r['variant']==name and int(r['iteration'])>1)
               for name in ('mapping/baseline','mapping/cached_lookup','mapping/rebuilt_lookup')}
analysis={'run_id':report['run_id'],'zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
    'checksums':verified,'manifest_matches':True,'recomputed_event_and_wall_medians':588,'verified_comparison_rows':384,
    'summary_groups':summaries,'focused_graph_results':focused,'setup_event_us_median_of_trial_medians':setup,
    'trace_kernel_sum_us_median_iterations_2_to_6':trace_summary,
    'environment':report['sections']['environment'],
    'decision':'Cached lookup merits an isolated integration/lifecycle review. Do not use per-call table reconstruction as the optimization. No serving or H200 gain has been established.',
    'statistics_note':'Descriptive aggregates of configuration/trial medians, not confidence intervals or independent H200 experiments. Percent reductions are computed within each matching trial before aggregation.'}
lines=['NOTEBOOK 03: VERIFIED MAPPING-CANDIDATE REVIEW','Run: '+report['run_id'],'',
    'DECISION','The cached lookup candidate has consistent isolated T4 evidence. Proceed to a narrow integration/lifecycle review, keeping active V4 unchanged until that review is complete. Rebuilding the table each call is not a robust optimization.','',
    'VALIDATION','All 11 member checksums and source provenance matched. Recomputed all 588 event AND wall medians from 6,468 samples; verified all 384 comparison rows.',
    '672 correctness cases / 4,032 output checks passed. All 96 benchmark cases, 1,152 changed-input benchmark graph checks and six setup graph checks passed.',
    'The profiler captured 114 actual kernels, all attributed. GPU annotation spans are excluded.','',
    'RESULTS: matched configuration/trial comparisons',
    'Scope | Candidate | Mode | wins/total | median reduction | reduction range']
for r in summaries:
    d=r['latency_reduction_percent']
    lines.append(f"{r['scope']} | {r['candidate']} | {r['execution']} | {r['event_median_wins']}/{r['comparisons']} | {d['median']:.2f}% | {d['min']:.2f}% to {d['max']:.2f}%")
lines += ['', 'FOCUSED GRAPH CASES: fallback cast+map, int64, hot-local',
          'Tokens | Baseline us* | Cached candidate us* | median paired trial reduction']
for r in focused:
    lines.append(f"{r['tokens']} | {r['baseline_us_median_of_trial_medians']:.2f} | {r['candidate_us_median_of_trial_medians']:.2f} | {r['median_within_trial_reduction_percent']:.2f}%")
lines += ['* Times are medians of three trial medians. Percentages are medians of within-trial percentage reductions, so they need not equal the ratio of these two aggregated times.','',
    'MECHANISM','At 128 tokens, int64 fallback input, each baseline invocation used seven kernels; cached lookup used five; rebuilding the table restored seven.',
    f"Separate profiler-on kernel-duration sums, median of iterations 2-6: baseline {trace_summary['mapping/baseline']:.2f} us, cached {trace_summary['mapping/cached_lookup']:.2f} us, rebuilt {trace_summary['mapping/rebuilt_lookup']:.2f} us. These trace times are not benchmark latency and cannot be subtracted from it.",
    'The cached fallback graph variant saved a median 5.63 us per isolated call across 48 configuration/trial comparisons (range about 4.77-7.42 us). This is more relevant than unusually large eager wins under host variability.','',
    'CACHE COSTS',
    f"Build new table: eager {setup['build_table_eager']:.2f} us, captured graph {setup['build_table_graph']:.2f} us. Refresh existing table via build+copy: eager {setup['refresh_existing_table_eager']:.2f} us, captured graph {setup['refresh_existing_table_graph']:.2f} us. All are medians of trial medians from separate setup measurements.",
    'Extra storage is 1,028 bytes per 256-entry int32 mapping. Cached timing excludes setup and refresh. Per-call rebuild gains are small and inconsistent: fallback eager wins only 28/48 comparisons, with a median reduction of 0.76%; fallback graph median reduction is 1.70%.',
    'Do not add separately measured stage medians into an exact break-even formula. Actual table lifetime, refresh frequency, graph use and device movement must be validated.','',
    'SOURCE REVIEW AND NEXT STEP',
    'MuseSpark1xFp4Experts creates _expert_map during initialization and registers it as a nonpersistent buffer (muse_spark_fp4_moe.py:170-177). The same file reads it in _marlin_expert_outputs; no other writes were found in that file. This suggests a reusable table is feasible, but is not proof against external updates.',
    'Prepare an isolated candidate that owns a companion nonpersistent lookup buffer. Verify device migration, buffer lifetime, arbitrary mapping content, checkpoint behavior, refresh/invalidation on mapping changes, and stable pointers during graph replay.',
    'Apply the lookup only where the standalone mapping fallback is already used. Keep the existing eligible <=64-route fused alignment path and its threshold unchanged. Include the standard dispatcher only if its separate ownership/lifecycle is reviewed.',
    'Test the integrated wrapper against baseline with repeated eager/graph measurements on T4. Preserve an independently reviewable patch and V4 baseline; do not claim a model speedup from this component result.','',
    'LIMITS',
    'This run used Tesla T4, torch 2.14.1+cu130, Triton 3.8.0 and driver 580.82.07. Its clock snapshot differs from Notebook 02; use within-run comparisons.',
    'Three trials are within one runtime/session, not independent hardware runs. Eager measurements show host variability; wins near zero in the rebuilt variant are not a strong basis for a change.',
    'Not measured: actual alignment/GEMMs, eight-GPU communication, full serving latency, scheduler/tokenizer startup, or H200 performance. No full-model correctness claim.', '']
out.mkdir(parents=True,exist_ok=True)
(out/'NOTEBOOK03_ANALYSIS.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
(out/'NOTEBOOK03_REVIEW.txt').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'checksums_verified':len(verified),'timing_rows_verified':len(timings),'comparison_rows_verified':len(comparisons),
                  'focused_graph_results':focused,'saved_to':str(out.resolve())},indent=2))
