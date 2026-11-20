import hashlib,json,datetime
from pathlib import Path
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Local SM86 component research, not H200 or serving qualification','target_serving_speedup':3.813,'confirmed_serving_score_unchanged':1.2237,'new_submissions':0,'runs':[],'notes':['Numerical gates unchanged. Original-policy negative reproduced exactly; K-preserving intervention passes tested cases.','Single-pass legal all-local unique routes can regress modestly. Duplicate robustness patterns are not realistic top-k distribution.','All-remote microcost is not EP critical-path cost.','Unsplit invariance old inherited descriptive labels are stale: harness adds a hard raise on baseline gate failure. Successful report establishes only tested local baseline and invariance conditions.']}
for folder in sorted(P.iterdir()):
    if not folder.is_dir() or not (folder/'run_receipt.json').exists():continue
    receipt=json.loads((folder/'run_receipt.json').read_text())
    row={'name':folder.name,'receipt_sha256':sha(folder/'run_receipt.json'),'exit_code':receipt.get('exit_code'),'command':receipt['command'],'executed_source_snapshots':receipt['source_snapshots']}
    rp=folder/'results.json'
    if rp.exists():
        data=json.loads(rp.read_text());row.update(status=data.get('status'),report_sha256=sha(rp),receipt_report_hash_matches=receipt.get('report_sha256')==sha(rp),failure=data.get('failure'))
        if 'metadata' in data:row['metadata_checks']=sum(v['checks'] for v in data['metadata'])
        if 'checks' in data:
            row['check_cases']=len(data['checks']);row['all_checks_pass']=all(v.get('passed',v.get('bitwise',False)) for v in data['checks'])
        if 'rows' in data and data['rows'] and 'reference_equal' in data['rows'][0]:
            row['sanitizer_reference_cases']=len(data['rows']);row['all_reference_cases_equal']=all(v['reference_equal'] for v in data['rows'])
        log=folder/'stdout.log'
        if log.exists():
            row['sanitizer_summary_lines']=[line for line in log.read_text(errors='replace').splitlines() if 'ERROR SUMMARY:' in line or 'RACECHECK SUMMARY:' in line]
        journal=folder/'comparison_journal.json'
        if journal.exists():
            comparisons=json.loads(journal.read_text());row.update(numerical_checks=len(comparisons),numerical_all_bitwise=all(v.get('bitwise',False) for v in comparisons),numerical_all_pass=all(v.get('passed',False) for v in comparisons))
        if 'fullchain' in data:row['timing']=[{k:v[k] for k in ('m','rank','cutoff_factor','timing_reference_factor','pattern','status','speedup','noise_pass','medians_us','controls') if k in v} for v in data['fullchain']]
        if 'baseline_tolerance_checks' in data:
            row['baseline_checks']=len(data['baseline_tolerance_checks']);row['baseline_gate_failures']=sum(not v.get('passed',False) for v in data['baseline_tolerance_checks']);row['baseline_all_bitwise']=all(v['bitwise'] for v in data['baseline_tolerance_checks']);row['strict_checks']=len(data['journal']);row['strict_all_bitwise']=all(v['bitwise'] for v in data['journal'])
    if not rp.exists():row['status']='no_results_file';row['qualification']='Incomplete; no numerical report produced'
    report['runs'].append(row)
(P/'EVIDENCE_SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'runs':len(report['runs']),'all_available_report_hashes_match':all(r.get('receipt_report_hash_matches',True) for r in report['runs'])}))

