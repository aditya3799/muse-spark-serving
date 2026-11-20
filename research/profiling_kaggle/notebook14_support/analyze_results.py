import csv,hashlib,io,json,statistics,zipfile
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=Path(r'D:\OpenKernelChallenge\serving\profiling_kaggle')
archive=ROOT/'notebook14_results/v4_cache_choice_20261114T200742Z_6317d4.zip'
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
        assert abs(float(r[metric+'_baseline_us'])-b)<1e-8 and abs(float(r[metric+'_candidate_us'])-c)<1e-8
        assert abs(float(r[metric+'_reduction_percent'])-100*(1-c/b))<1e-8
        assert abs(float(r[metric+'_saved_us'])-(b-c))<1e-8
        assert abs(float(r[metric+'_saved_us_per_stage'])-(b-c)/depth)<1e-8
assert len(devices)==672 and all(d['category']=='kernel' and d['variant']!='unattributed' for d in devices)
assert len(trace)==18
for r in trace:assert (int(r['kernel_count']),int(r['copy_memset_count']))==(48 if '/baseline_' in r['variant'] else 32,0)
def summary(rows,metric):
    vals=[float(r[metric+'_reduction_percent']) for r in rows]
    return {'wins':sum(v>0 for v in vals),'comparisons':len(vals),'median_reduction_percent':statistics.median(vals),'min_reduction_percent':min(vals),'max_reduction_percent':max(vals),'median_saved_us_per_stage':statistics.median(float(r[metric+'_saved_us_per_stage']) for r in rows)}
overall=[]
for candidate,baseline in sorted(set((r['candidate'],r['baseline']) for r in comparisons)):
    selected=[r for r in comparisons if (r['candidate'],r['baseline'])==(candidate,baseline)]
    overall.append(dict(candidate=candidate,baseline=baseline,**{m:summary(selected,m) for m in ('event','wall','enqueue')}))
previous=json.loads((ROOT/'notebook11_results/NOTEBOOK11_ANALYSIS.json').read_text())
old={(r['candidate'],r['baseline']):r for r in previous['overall']}
new={(r['candidate'],r['baseline']):r for r in overall}
replication=[]
for schedule in ('eager','graph'):
    pair=('two_buffer_'+schedule,'baseline_'+schedule)
    replication.append({'schedule':schedule,'earlier_colab_reduction_percent':old[pair]['event']['median_reduction_percent'],
        'latest_colab_reduction_percent':new[pair]['event']['median_reduction_percent'],'latest_wins':new[pair]['event']['wins'],'comparisons':48})
analysis={'run_id':report['run_id'],'zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'member_checksums_verified':len(hashes),
    'provider':'Google Colab T4, explicitly confirmed by user','intended_kaggle_replication_completed':False,
    'timing_rows_recomputed_all_three_metrics':288,'comparison_rows_verified':288,'overall':overall,'replication':replication,
    'environment':report['sections']['environment'],'decision':'Close this MoE source-slice T4 profiling phase after Notebook 14. Two-buffer remains selected, frozen and unapplied. No Notebook 15 or forced Kaggle repetition; unresolved integration and platform questions are documented.'}
lines=['NOTEBOOK 14: VERIFIED COLAB REPEAT','Run: '+report['run_id'],'',
    'PROVIDER AND INTEGRITY',
    'User explicitly confirmed execution on Colab. The notebook filename requested Kaggle, but it is not evidence of execution provider. Runtime reports PyTorch 2.14.1+cu130, Triton 3.8.0, NumPy 2.4.6 and driver 580.82.07 on one visible Tesla T4.',
    f'All {len(hashes)} payload hashes and the original Notebook 11 source manifest match. All 48 cases and 2,592 changed-input output checks pass. Recomputed 288 rows from 3,168 samples and verified all 288 comparisons across event, wall and enqueue metrics.',
    'Trace: 672 kernels, no copies, no unattributed operations. Eight-stage baseline uses 48 kernels, both candidates 32.','',
    'MATCHED RESULTS',
    'Candidate / Reference | Event reduction (wins/48) | Wall reduction | Event savings per stage (us)']
for r in overall:lines.append(f"{r['candidate']} / {r['baseline']} | {r['event']['median_reduction_percent']:.2f}% ({r['event']['wins']}/48) | {r['wall']['median_reduction_percent']:.2f}% | {r['event']['median_saved_us_per_stage']:.3f}")
lines+=['Negative reduction means slower; descriptive paired medians, not confidence intervals.','',
    'REPEATABILITY ON COLAB',
    'Schedule | Earlier Colab Notebook 11 | Latest Colab Notebook 14']
for r in replication:lines.append(f"{r['schedule']} | {r['earlier_colab_reduction_percent']:.2f}% | {r['latest_colab_reduction_percent']:.2f}%")
lines+=['Both runs favor two-buffer under static maps. Latest two-buffer wins 48/48 in both schedules. Single-owner is 11.80% slower than two-buffer in eager execution and 0.55% slower at the graph median; the small graph difference is not proof of universal equivalence.',
    'This repeats the Colab conclusion. It does not measure Kaggle latency and does not produce an H200/full-serving speedup.','',
    'STOPPING POINT',
    'This is the final planned MoE notebook in the current phase. Checks passed, so no new notebook is generated. The intended Kaggle performance replication remains unmeasured; it is documented rather than used to extend the loop.',
    'The unchanged two-buffer draft remains the preferred research candidate for static/sufficiently reused maps. It is not applied to active V4. See MOE_MAPPING_FINAL_REVIEW.txt for consolidated evidence and remaining integration work.','']
(HERE/'NOTEBOOK14_ANALYSIS.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
(HERE/'NOTEBOOK14_REVIEW.txt').write_text('\n'.join(lines),encoding='utf-8')

selected=json.loads((ROOT/'notebook12_support/source_manifest_12.json').read_text())
decision={'phase':'MoE mapping source-slice profiling','status':'closed_after_notebook14','date_local':'2026-11-15','next_notebook_planned':False,
    'selected_draft':'notebook12_support/selected_two_buffer_v4.diff','selected_diff_sha256':selected['selected_diff_sha256'],
    'selected_source_sha256':selected['selected_source_sha256'],'active_v4_patch_sha256':selected['baseline_patch_sha256'],'applied_to_active_v4':False,
    'condition':'Static/sufficiently reused maps; every map mutation must refresh lookup before consumption. Preserve graph storage or recapture after replacement/moves.',
    'latest_colab_results':replication,'kaggle_performance_replication_measured':False,
    'remaining':['Actual fused small-input alignment and expert GEMMs','Full serving/checkpoint/buffer integration and all map writers','Actual runtime router layouts/backend','Kaggle performance transfer','H200/full-serving latency','Scheduler/tokenizer initialization profiling']}
(HERE/'MOE_MAPPING_DECISION.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
final=['MOE MAPPING: PHASE CLOSEOUT','Status: closed after Notebook 14; no further notebook scheduled.','',
    'DECISION',
    'Retain the unchanged two-buffer draft as the preferred research candidate for static/sufficiently reused maps. Keep active V4 unchanged. This is selection for possible integration, not a claim of deployment readiness.',
    'Selected diff: notebook12_support/selected_two_buffer_v4.diff',
    'Diff SHA256: '+selected['selected_diff_sha256'],
    'Selected full source: notebook12_support/selected_two_buffer_source.py',
    'Source SHA256: '+selected['selected_source_sha256'],'',
    'WHAT WAS MEASURED',
    'The pre-alignment portion of MuseSpark1xFp4Experts: expert-ID conversion, global-to-local mapping and score conversion. Tests execute AST-selected constructor/prefix code with synthetic inputs, not the full class or model.',
    'Original map lookup is compared with a cached lookup containing a leading drop sentinel. Two-buffer retains the original registered map; single-owner makes the map a property view into lookup storage.',
    'For native int32 IDs/float32 scores, baseline uses six kernels per measured prefix and both caches use four in the traced cases. These counts concern this prefix, not entire MoE or model layers.','',
    'PERFORMANCE EVIDENCE',
    'Latest Colab repeat: two-buffer median event latency reduction 30.08% eager and 25.73% graph, 48/48 wins in each. Earlier equivalent Colab run: 30.06% and 24.91%. Matched within-run comparisons support repeatability of this narrow result.',
    'Single-owner has no measured advantage worth preferring for this static workload: latest eager latency is 11.80% higher than two-buffer, graph median 0.55% higher. Small graph differences should not be overinterpreted.',
    'Earlier int64/float16 harnesses included conversions absent from inspected router-default int32/float32 outputs. Notebook 10 isolated that effect; percentages across input contracts are not additional code optimizations.',
    'Frequent eager refresh can erase or reverse cached-mapping gains. There is no unconditional 25-30% serving improvement and no universal refresh crossover established.','',
    'CORRECTNESS EVIDENCE',
    'Notebook 12 on Kaggle: 1,296 configurations; 2,208 output-value checks plus 384 metadata-only small fused-branch checks; 128 graph-update outputs; 48 device-move/recapture outputs; 25 invalid updates rejected.',
    'Notebook 13 on Kaggle: 768 explicitly event-ordered cross-stream handoffs, 1,536 outputs, 768 stable-pointer checks and eight recaptures after moves passed. Host validation waits each cycle; not concurrent-request throughput.',
    'Notebook 14 on Colab: all lifecycle, input, preservation, benchmark and trace checks pass. These experiments do not execute the omitted small fused aligner or expert GEMMs.','',
    'PLATFORM PROVENANCE',
    'User confirmed earlier timing experiments and Notebook 14 ran on Colab T4. Notebooks 12 and 13 ran on Kaggle T4 after Colab limits were exhausted.',
    'Recent Colab: PyTorch 2.14.1+cu130 / Triton 3.8.0. Kaggle correctness runs: PyTorch 2.11.0+cu130 / Triton 3.6.0. Notebook filenames do not establish provider.',
    'Kaggle performance replication was intended for Notebook 14 but was not performed there. This is an explicit evidence gap, not a reason to extend the agreed notebook loop.','',
    'INTEGRATION REQUIREMENTS',
    'Every direct map edit must refresh lookup before consumers, or use the checked update API. Two-buffer stale output was deliberately demonstrated and fixed with refresh.',
    'Updates on a different CUDA stream need a ready-event dependency before consumer replay and completion ordering before reuse. The harness demonstrates a correct schedule; production call sites still require audit.',
    'Moving/replacing buffers requires fresh graph capture. Real model loading, generic buffer manipulation and checkpoint integrations must preserve the map/lookup relation.',
    'Actual small fused alignment and GEMMs need supported-hardware testing. H200/full-serving latency remains unknown; T4 prefix timing is not an end-to-end estimate.','',
    'OUTSIDE THIS PHASE',
    'Scheduler initialization and tokenizer initialization were not profiled. They require a separately scoped plan if the user chooses that work. Do not automatically start another profiling phase or generate Notebook 15.',
    'No new submissions, model deployments or active V4 source edits were made in closing this phase.','']
(HERE/'MOE_MAPPING_FINAL_REVIEW.txt').write_text('\n'.join(final),encoding='utf-8')
print(json.dumps({'verified_checksums':len(hashes),'replication':replication,'decision':decision['status'],'new_notebook':False},indent=2))
