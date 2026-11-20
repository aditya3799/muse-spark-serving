import csv,hashlib,io,itertools,json,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=Path(r'D:\OpenKernelChallenge\serving\profiling_kaggle')
archive=ROOT/'notebook12_results/results.zip'
with zipfile.ZipFile(archive) as outer:
    assert len(outer.namelist())==len(set(outer.namelist()))
    nested=[n for n in outer.namelist() if n.endswith('.zip')];assert len(nested)==1
    inner_bytes=outer.read(nested[0])
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as z:
        assert len(z.namelist())==len(set(z.namelist()))
        hashes=json.loads(z.read('checksums.json'));assert set(hashes)==set(z.namelist())-{'checksums.json'}
        for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
        prefix=nested[0][:-4]+'/'
        for n in z.namelist():assert z.read(n)==outer.read(prefix+n),n
        report=json.loads(z.read('report.json'))
        def table(n):return list(csv.DictReader(io.StringIO(z.read(n).decode())))
        matrix,updates,moves=map(table,('matrix_checks.csv','graph_update_checks.csv','move_checks.csv'))
assert report['source_manifest']==json.loads((ROOT/'notebook12_support/source_manifest_12.json').read_text())
assert report['gate_status']=='passed' and all(s['status']=='passed' for s in report['sections'].values())
assert (len(matrix),len(updates),len(moves))==(1296,64,8)
assert all(r['status']=='passed' for rows in (matrix,updates,moves) for r in rows)
keys=('rank','tokens','id_dtype','strided','pattern')
actual={tuple(r[k] for k in keys) for r in matrix}
expected=set(itertools.product([str(i) for i in range(8)]+['no_map'],map(str,(0,1,2,8,9,16,64,128,512)),('int32','int64'),('False','True'),('local','remote','mixed','negative')))
assert actual==expected and len(actual)==len(matrix)
for r in matrix:
    fused=r['rank']!='no_map' and r['strided']=='False' and int(r['tokens']) in (1,2,8)
    assert (r['fused_deferred']=='True')==fused
assert sum(r['fused_deferred']=='True' for r in matrix)==192
assert {tuple(r[k] for k in ('rank','path','mapping')) for r in updates}==set(itertools.product(map(str,range(8)),('checked_update','direct_then_refresh'),('initial','all_remote','reversed','next_partition')))
assert all(r['poisoned_state_replaced']=='True' for r in updates)
assert {r['rank'] for r in moves}==set(map(str,range(8))) and sum(int(r['output_checks']) for r in moves)==48
s=report['sections'];assert s['matrix']['outputs_checked']==2592 and s['matrix']['value_checked_outputs']==2208 and s['matrix']['deferred_fused_metadata_outputs']==384
assert s['updates']['graph_output_checks']==128 and s['updates']['stable_pointer_checks']==64
assert s['contracts']['invalid_updates_rejected_before_mutation']==25 and s['contracts']['independent_consistent_deepcopies']==8
previous=json.loads((ROOT/'notebook11_results/NOTEBOOK11_ANALYSIS.json').read_text())
analysis={'run_id':report['run_id'],'outer_zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'nested_zip':nested[0],
    'nested_zip_sha256':hashlib.sha256(inner_bytes).hexdigest(),'verified_payload_checksums':len(hashes),'nested_and_expanded_files_identical':True,
    'complete_matrix_cross_product_verified':True,'matrix_cases':1296,'value_checked_outputs':2208,'metadata_only_outputs':384,'graph_update_outputs':128,'move_outputs':48,
    'environment':s['environment'],'invalid_updates_rejected':25,
    'decision':'Selected unchanged two-buffer source-slice gate passes on the new Kaggle stack. Keep draft unapplied. Next validate explicit event ordering across producer/consumer CUDA streams; no new latency claim.'}
lines=['NOTEBOOK 12: VERIFIED TWO-BUFFER GATE','Run: '+report['run_id'],'',
    'ARCHIVE INTEGRITY',f'Input is results.zip containing {nested[0]} and an expanded copy. Verified all {len(hashes)} payload hashes; every expanded member exactly matches the inner ZIP. Provenance matches the frozen Notebook 12 source manifest.','',
    'CORRECTNESS',
    'All 1,296 matrix configurations are present exactly once: eight partitions plus no-map, nine token counts, two ID dtypes, two layouts and four routing patterns.',
    '2,208 outputs passed full ID-value checks. Another 384 outputs reached the small fused branch and passed metadata/dispatch checks only; their uninitialized ID values were not read.',
    'All 64 changed-map cycles passed, including 128 baseline/candidate graph outputs and 64 stable-pointer checks. Poisoned old contents were replaced before validation.',
    'All eight GPU -> CPU -> GPU -> recapture sequences passed, with 48 output checks. Old graphs were not replayed after relocation.',
    'All 25 malformed/no-map updates were rejected before mutation. Eight independent deep copies and marker-only state_dict/checkpoint checks passed.','',
    'ENVIRONMENT CHANGE',
    'This run: PyTorch 2.11.0+cu130, Triton 3.6.0, NumPy 2.1.3, driver 580.178.04. Two Tesla T4 devices were visible; only device 0 was active for this notebook.',
    'Recent timing notebooks used PyTorch 2.14.1+cu130 and Triton 3.8.0. The new run adds correctness compatibility evidence on a different software stack. It does not reproduce previous speedups: Notebook 12 contains no latency benchmark and its section durations are not performance results.','',
    'DECISION',
    'The unchanged two-buffer draft passes the expanded source-slice correctness gate. It remains selected for static/sufficiently reused maps, and remains unapplied to active V4.',
    'Refresh discipline still matters: a direct map write without refreshing the lookup can produce stale outputs. Production map writers and generic buffer replacement/loading paths still need full integration review.',
    'Actual fused small-input alignment/GEMMs, full model loading, H200 performance and scheduler/tokenizer initialization remain outside the evidence.','',
    'NEXT: 13_kaggle_stream_ordering.ipynb',
    'Test safe cross-stream execution on GPU 0: producer updates map/lookup and routing inputs, records an event; consumer waits on that event before replay. A completion event fences output checks and buffer reuse.',
    'Cover eight partitions, both checked-update and explicit-refresh APIs, default/dedicated consumer streams, repeated changed maps, and graph recapture after a device round trip.',
    'No deliberately unsynchronized race is used. Passing proves the tested event-ordered schedule only; it does not establish that all production call sites obey this schedule. No timing benchmark or source algorithm change is introduced.','']
(HERE/'NOTEBOOK12_ANALYSIS.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
(HERE/'NOTEBOOK12_REVIEW.txt').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({k:v for k,v in analysis.items() if k!='environment'},indent=2))
