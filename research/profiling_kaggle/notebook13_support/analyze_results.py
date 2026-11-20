import csv,hashlib,io,itertools,json,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=Path(r'D:\OpenKernelChallenge\serving\profiling_kaggle')
archive=ROOT/'notebook13_results/results.zip'
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
        smoke,handoffs=map(table,('smoke_checks.csv','stream_handoff_checks.csv'))
assert report['source_manifest']==json.loads((ROOT/'notebook13_support/source_manifest_13.json').read_text())
assert report['gate_status']=='passed' and all(s['status']=='passed' for s in report['sections'].values())
assert len(smoke)==16 and all(r['status']=='passed' for r in smoke)
assert {(r['rank'],r['variant']) for r in smoke}==set(itertools.product(map(str,range(8)),('baseline','two_buffer')))
keys=('rank','phase','consumer','update_path','round','mapping')
expected=set(itertools.product(map(str,range(8)),('initial_capture','recapture_after_move'),('default','dedicated'),('checked_update','direct_then_refresh'),map(str,range(3)),('initial','all_remote','reversed','next_partition')))
assert len(handoffs)==768 and {tuple(r[k] for k in keys) for r in handoffs}==expected
assert sum(int(r['output_checks']) for r in handoffs)==1536
for r in handoffs:
    assert r['status']=='passed' and r['producer_stream_handle']!=r['consumer_stream_handle']
    assert all(r[k]=='True' for k in ('ready_wait_used','completion_event_wait_used','stable_pointers'))
s=report['sections'];h=s['handoff']
assert (h['handoff_cycles'],h['output_checks'],h['stable_pointer_checks'],h['partitions_recaptured_after_move'])==(768,1536,768,8)
assert not h['device_wide_sync_in_handoff'] and h['intentionally_unsynchronized_cases']==0
assert s['contracts']['invalid_updates_rejected_before_mutation']==25
analysis={'run_id':report['run_id'],'outer_zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'nested_zip':nested[0],
    'nested_zip_sha256':hashlib.sha256(inner_bytes).hexdigest(),'verified_payload_checksums':len(hashes),'nested_and_expanded_files_identical':True,
    'complete_handoff_cross_product_verified':True,'handoff_cycles':768,'output_checks':1536,'stable_pointer_checks':768,'recaptured_partitions':8,
    'environment':s['environment'],'provider':'Kaggle T4 (ongoing user-confirmed workflow; earlier timing runs were Colab T4)',
    'decision':'Event-ordered correctness gate passed. Keep the selected two-buffer draft unapplied. Re-run the exact Notebook 11 timing workload on Kaggle before transferring the Colab performance conclusion.'}
lines=['NOTEBOOK 13: VERIFIED STREAM-ORDERING REVIEW','Run: '+report['run_id'],'',
    'INTEGRITY',f'Input results.zip contains {nested[0]} and its expanded files. All {len(hashes)} payload checksums pass, expanded and inner files are identical, and source provenance matches Notebook 13.','',
    'GATE RESULTS',
    'All 768 distinct handoffs are present exactly once: 8 partitions x 2 capture phases x 2 consumer stream types x 2 update APIs x 3 rounds x 4 map states.',
    'All 1,536 baseline/candidate outputs and 768 stable-pointer checks passed. Producer and consumer handles differ in every row, and each row records ready-event and completion-event waits.',
    'Both initial capture and recapture after a device round trip passed for all eight partitions. Both checked updates and explicit map-copy-plus-refresh passed with default and dedicated consumer streams.',
    'All 16 smoke outputs and 25 invalid-update rejection checks passed.','',
    'WHAT THIS ESTABLISHES',
    'The tested schedule correctly orders producer writes before consumer graph replay, then waits for completion before checking output and reusing buffers. Setup/teardown can synchronize the device; no device-wide synchronization is used in the handoff itself.',
    'The host waits after each cycle. This is not a multi-request throughput test or proof about arbitrary concurrent writers. No unsynchronized race was run, and no production call site was changed.',
    'Actual fused alignment/GEMMs, full serving/checkpoint/buffer integration, runtime router layouts and H200 behavior remain untested. The selected two-buffer draft remains unapplied; active V4 is unchanged.','',
    'PLATFORM PROVENANCE',
    'This is the current Kaggle T4 workflow, per the user\'s clarification. Runtime reports PyTorch 2.11.0+cu130, Triton 3.6.0, NumPy 2.1.3 and driver 580.178.04. Two T4 devices are visible; only GPU 0 is active.',
    'The earlier cache timing experiments were Colab T4, including Notebook 11 on PyTorch 2.14.1+cu130. Notebook 12 and 13 add correctness evidence on Kaggle, but contain no latency benchmark. Their short section durations are not speedups.','',
    'NEXT: 14_kaggle_cache_replication.ipynb',
    'Re-run the exact executable cells of Notebook 11 on the current Kaggle environment: baseline, two-buffer and single-owner; eager and graph; int32 IDs/float32 scores; 9/32/128/512 tokens; 1/8 stages; hot-local/uniform routes; three trials.',
    'Executable code, configuration and embedded source manifest are unchanged. Notebook 14 changes only explanatory Markdown and output-folder instructions. It intentionally retains the v4_cache_choice ZIP prefix and original result schema so the same analysis applies.',
    'Expected 48 cases, 288 timing rows, 3,168 samples, 2,592 changed-input output checks and 288 comparisons. Compare baseline/candidate within this Kaggle run. Cross-provider conclusions should compare matched ratios and directions; do not attribute raw absolute-time changes to code alone.',
    'No local GPU re-execution is needed for this Markdown-only replication artifact: its code is byte-for-byte identical to the previously GPU-validated Notebook 11, whose artifact hash and validation record are verified. The Kaggle run is the new experiment.','']
(HERE/'NOTEBOOK13_ANALYSIS.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
(HERE/'NOTEBOOK13_REVIEW.txt').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({k:v for k,v in analysis.items() if k!='environment'},indent=2))
