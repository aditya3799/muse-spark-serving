import csv,hashlib,io,json,statistics,zipfile
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\OpenKernelChallenge\serving\profiling_kaggle')
archive=next(ROOT.rglob('v4_router_contract_20261114T182013Z_2a6191.zip'))
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==len(set(z.namelist()))
    hashes=json.loads(z.read('checksums.json'));assert set(hashes)==set(z.namelist())-{'checksums.json'}
    for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    report=json.loads(z.read('report.json'))
    def table(n):return list(csv.DictReader(io.StringIO(z.read(n).decode())))
    timings,raw,cases,comparisons,trace,devices=map(table,('timings.csv','raw_samples.csv','benchmark_cases.csv','comparisons.csv','trace_variants.csv','trace_device_operations.csv'))
assert report['source_manifest']==json.loads((ROOT/'notebook10_support/source_manifest_10.json').read_text())
assert all(r['status']=='passed' for r in report['sections'].values())
assert (len(timings),len(raw),len(cases),len(comparisons))==(144,1584,18,252)
assert all(c['status']=='passed' for c in cases)
assert sum(int(c['changed_input_output_checks']) for c in cases)==1296
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
        for label,value in [('baseline',b),('candidate',c)]:
            assert abs(float(r[metric+'_'+label+'_us'])-value)<1e-8
            assert abs(float(r[metric+'_'+label+'_us_per_stage'])-value/depth)<1e-8
        assert abs(float(r[metric+'_reduction_percent'])-100*(1-c/b))<1e-8
assert len(devices)==1152 and all(d['category']=='kernel' and d['variant']!='unattributed' for d in devices)
assert len(trace)==24
for r in trace:
    name=r['variant'];base=6 if '/baseline_' in name else 4
    extra=int('i64' in name)+int('s16' in name)
    assert (int(r['kernel_count']),int(r['copy_memset_count']))==((base+extra)*8,0)
def summary(rows,metric):
    vals=[float(r[metric+'_reduction_percent']) for r in rows]
    return {'wins':sum(v>0 for v in vals),'comparisons':len(vals),'median_reduction_percent':statistics.median(vals),'min_reduction_percent':min(vals),'max_reduction_percent':max(vals)}
overall=[];native_groups=[]
for candidate,baseline in sorted(set((r['candidate'],r['baseline']) for r in comparisons)):
    selected=[r for r in comparisons if (r['candidate'],r['baseline'])==(candidate,baseline)]
    overall.append(dict(candidate=candidate,baseline=baseline,**{m:summary(selected,m) for m in ('event','wall','enqueue')}))
native=[r for r in comparisons if r['candidate']=='single_i32_s32' and r['baseline']=='baseline_i32_s32']
for n in (9,32,128):
    for d in (1,8):native_groups.append(dict(tokens=n,stages=d,**{m:summary([r for r in native if r['tokens']==str(n) and r['stages']==str(d)],m) for m in ('event','wall','enqueue')}))
analysis={'run_id':report['run_id'],'zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'member_checksums_verified':len(hashes),'timing_rows_recomputed_all_three_metrics':144,'comparison_rows_verified':252,'overall':overall,'native_contract_by_shape':native_groups,
    'decision':'Candidate wins under inspected default output dtypes. Next compare the two existing ownership designs using native dtypes and static maps, before selecting an integration draft. No source promotion.'}
lines=['NOTEBOOK 10: VERIFIED ROUTER-CONTRACT REVIEW','Run: '+report['run_id'],'',
    'VALIDATION',f'All {len(hashes)} member checksums and source provenance match. Recomputed all metrics for 144 rows from 1,584 samples; verified 252 comparisons. All 18 cases, 1,296 changed-input output checks, 80 dtype-contract output checks and inherited 576 output checks passed.',
    'Trace: 1,152 kernels; no copies and no unattributed operations.','',
    'MATCHED REDUCTIONS (%)',
    'Candidate / Reference | Event median (wins/18) | Wall median | Enqueue median']
for r in overall:lines.append(f"{r['candidate']} / {r['baseline']} | {r['event']['median_reduction_percent']:.2f}% ({r['event']['wins']}/18) | {r['wall']['median_reduction_percent']:.2f}% | {r['enqueue']['median_reduction_percent']:.2f}%")
lines+=['Negative reductions mean slower. Descriptive within-run ratios, not confidence intervals.','',
    'NATIVE CONTRACT BY SHAPE: candidate versus baseline, int32 IDs / float32 scores',
    'Tokens | Stages | Event median | Range | Wins/3']
for r in native_groups:lines.append(f"{r['tokens']} | {r['stages']} | {r['event']['median_reduction_percent']:.2f}% | {r['event']['min_reduction_percent']:.2f}% to {r['event']['max_reduction_percent']:.2f}% | {r['event']['wins']}/3")
lines+=['','FINDINGS',
    'Under int32 IDs / float32 scores, single-owner versus baseline wins 18/18: 26.06% median lower CUDA-event latency, 21.79% lower synchronized wall time. Historical int64/float16 inputs yield a 17.87% event reduction in this same run.',
    'Trace per stage: historical baseline/candidate = 8/6 kernels; native-contract baseline/candidate = 6/4. Each unnecessary dtype conversion adds one kernel to both implementations. The mapping difference remains two kernels.',
    'Matching default dtypes lowers baseline harness event latency by 26.54% and candidate latency by 33.70% versus the historical inputs. These are benchmark-input effects, not new serving optimizations. The larger native-contract percentage partly reflects the smaller denominator after removing common conversion work.',
    'The source audit identifies default JIT/Triton output allocations. Actual router execution, preallocated buffers, runtime layout and backend selection remain unobserved. All timed inputs are contiguous.',
    'The local judge launch specification in RULES.md includes --ep-size 8. The serving constructor creates an expert map when local and global expert counts differ; the tested partitioned-map scenario is therefore relevant to that configuration, but is not an observation of a running model.','',
    'NEXT: 11_kaggle_cache_choice.ipynb',
    'Compare baseline, earlier two-buffer cache and single-owner cache with int32 IDs / float32 scores, static maps, eager execution and combined graphs. Expand to 512 tokens and two routing distributions while retaining 1/8 independent stages.',
    'Two-buffer preserves the original registered _expert_map at offset zero, but needs explicit refresh after map edits and holds a second buffer. Single-owner prevents stale duplicated contents, but changes _expert_map to a property with a four-byte offset and requires integration review.',
    'If static performance is comparable, preserving the original map contract may be preferable. No equivalence or winner is assumed; report paired absolute and percentage differences. A separate gate demonstrates the two-buffer stale-copy behavior and verifies explicit refresh before graph replay.',
    'No actual alignment/GEMM, small fused kernel, communication, full model, scheduler/tokenizer startup or H200 validation. Neither draft has been applied; active V4 is unchanged.','']
(HERE/'NOTEBOOK10_ANALYSIS.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
(HERE/'NOTEBOOK10_REVIEW.txt').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'verified_members':len(hashes),'native_contract':summary(native,'event'),'native_by_shape':native_groups},indent=2))
