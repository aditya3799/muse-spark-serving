"""Summarize the same-kernel launch-grid experiment, with conservative decisions."""
from pathlib import Path
import hashlib,json,statistics
P=Path(__file__).resolve().parent
run=P/'grid_body1_retry'
r=json.loads((run/'results.json').read_text())
receipt=json.loads((run/'run_receipt.json').read_text())
assert r['status']=='completed' and receipt['exit_code']==0
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert receipt['report_sha256']==sha(run/'results.json')
rows=[]
for row in r['rows']:
    for v in row['variants']:
        assert all(v['numerical'].values()) and v['changed_input_graph_cases']==3
        rows.append({'m':row['m'],'pattern':row['pattern'],'active_routes':row['active_routes'],
                     'grid_cta_multiplier':v['grid_sms']/r['sm_count'],'grid_sms_argument':v['grid_sms'],
                     'warm_speedup':v['timing']['warm']['speedup'],'cold_speedup':v['timing']['cold']['speedup'],
                     'warm_AA':v['timing']['warm']['AA_ratio'],'cold_AA':v['timing']['cold']['AA_ratio'],
                     'controls_pass':all(t['AA_pass'] for t in v['timing'].values())})
telemetry=[json.loads(s) for s in (run/'loaded_telemetry.jsonl').read_text().splitlines()]
loaded=[]
for t in telemetry:
    fields=t.get('stdout','').strip().split(', ')
    if len(fields)==12 and int(fields[10])>0:
        loaded.append({'sm_MHz':int(fields[3]),'memory_MHz':int(fields[4]),'watts':float(fields[5]),'power_limit_W':float(fields[7]),'temperature_C':float(fields[8]),'utilization_percent':int(fields[10])})
result={'status':'REJECT_SMALLER_GRID_NO_MATERIAL_LARGER_GRID_GAIN','scope':r['scope'],
        'physical_sm_count':r['sm_count'],'grid_not_physical_resource_change':True,
        'static_raw_final_bitwise_checks':2*len(rows),'changed_input_graph_cases':3*len(rows),
        'control_failures':[x for x in rows if not x['controls_pass']],
        'rows':rows,'loaded_telemetry':loaded,
        'decision':'Do not promote a grid-size change. Active quarter/half grids regress; doubled grid has no substantial repeat-worthy benefit. Empty-route warm timing noise cannot justify policy.',
        'source_receipts':{str(p.relative_to(P)):sha(p) for p in [run/'results.json',run/'run_receipt.json',run/'source_snapshots/grid_probe.py',run/'loaded_telemetry.jsonl']},
        'preserved_failure':'grid_body1_pilot exited1 because harness requested hot rather than hot_expert; no original evidence overwritten.',
        'remote_submissions':0}
(P/'GRID_FINDINGS.json').write_text(json.dumps(result,indent=2)+'\n')
print('variants',len(rows),'AA failures',len(result['control_failures']),'static checks',result['static_raw_final_bitwise_checks'],'graph cases',result['changed_input_graph_cases'])
print('active rows:',len([x for x in rows if x['active_routes']]))
