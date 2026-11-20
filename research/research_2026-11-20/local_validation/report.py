import datetime, hashlib, json, pathlib
R=pathlib.Path('D:/OpenKernelChallenge/serving')
O=R/'research_2026-11-20/local_validation'
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else None
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
matrix=read(O/'matrix_state.json');factory=read(O/'factory_state.json')
records=[]
for job in matrix['jobs']:
    name=job['name'];parts=name.split('_')
    if name.startswith('adapter_'): p=O/f'adapter_probe_{parts[1]}_{parts[2]}.json'
    elif name.startswith('row_chain_'):p=O/(name+'.json')
    else:p=R/f'research_2026-11-19/decode/output_shard_{int(parts[2])+int(parts[3])}/result.json'
    data=read(p)
    item={**job,'evidence':str(p),'evidence_exists':data is not None}
    if data:
        item.update(result_complete=data.get('complete'),correct=data.get('correct'),scope=data.get('scope'))
        if name.startswith('adapter_'):
            item['shapes']=[dict(rows=c['rows'],enabled=c['enabled'],changed_graph_exact=c.get('changed_graph_exact'),fallback_exact=c.get('fallback_exact'),mismatch_count=sum(c['mismatches'])) for c in data['cases']]
            item['peak_allocated_bytes']=data.get('peak_allocated')
    active=None
    if data:
        if name.startswith('adapter_'):active=all(c.get('enabled') for c in data['cases'])
        elif name.startswith('row_chain_'):active=all(c.get('chosen') is not None for c in data['cases'])
        else:active=data.get('active')
    item['optimized_path_active']=active
    item['passed']=job.get('returncode')==0 and bool(data and data.get('complete') and data.get('correct')) and active is True
    records.append(item)
cpu={name:read(O/(name+'.json')) for name in ('integration_contract','output_n_actual_gloo','row_adapter_cpu_tp2','row_adapter_cpu_tp4','row_adapter_cpu_tp8','row_policy_gate','scheduler_stream_contract','live_shape_contract','watchdog_test')}
factory_results={name:read(O/(name+'.json')) for name in ('row_package_probe','output_n_package_probe_v2')}
patches={n:digest(R/'research_2026-11-19_final/candidates'/n/'candidate.diff') for n in ('row_only','decode_only','joint','disabled_control')}
jobs={i:read(O/f'status_{i}.json') for i in ('f464d5f09f534363','f43fa02892254879','f038a667401a4d0a','0afdc0b191dd4b42')}
report=dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    conclusion='Local correctness evidence only. No full serving or H200 speedup established. Preserve scored 1.2363x baseline pending target validation.',
    baseline_sha256=digest(R/'release_2026-11-18/candidate.diff'),candidate_sha256=patches,
    matrix_complete=matrix['complete'],matrix=records,factory_state=factory,factory_results=factory_results,cpu=cpu,official_jobs=jobs,
    post_state=read(O/'post_state.json'),large_shape_completions={str(seed):read(O/f'row_chain_completion_{seed}_16384.json') for seed in (20112023,20112029)},
    sanitizer_results={p.name:read(p) for p in (O/'sanitizer').glob('adapter_probe_*.json')},
    composition_state=read(O/'composition_state.json'),composition_results={str(m):read(O/f'exported_chain_{m}.json') for m in (8,256,512,3072)},
    tested_source_identity=read(O/'tested_source_identity.json'),
    telemetry=read(O/'telemetry_summary.json'),capture_scope_audit=read(O/'capture_scope_audit.json'),performance_promotion_supported=False,
    old_check_interpretation='0afdc0b191dd4b42 is the retired Hopper routed-expert patch 1ce48266..., not the current projection candidates. Correctness passed, but throughput was 0.529x stock; TTFT ratios 2.14x first and 7.91x later exceeded the 2x limit. It confirms the prior regression; no promotion.',
    limitations=['RTX3050 6GB SM86, not H200 SM90.', 'GPU observed at 210 MHz core, 405 MHz memory and approximately 10 W under load; timings unsuitable for serving projection.',
    'Eight GPU peers are serialized or mocked; CPU Gloo transport is real but is not GPU NCCL.',
    'Component chains include real FC weights and normalization/KV operations; the complete 816B model and target/draft blocks cannot run locally.',
    'Calibration plus held-out bit comparisons is empirical evidence, not proof of equivalence for every possible input.',
    'New MoE converter is standalone, excluded from all candidate patches; converter speedup is not MoE inference speedup.',
    'Six GPU adapter arms switch the same exported implementation on/off; they are not six full serving deployments. Candidate patch differences and startup hooks are checked separately.',
    'The three queued checks are immutable earlier submissions; their results will not automatically validate the later final candidates.'],
    uniform_improvement_scenario_for_2x=dict(current_verified_score=1.2363,target_score=2.0,required_time_reduction_vs_current=1-1.2363/2,explanation='Illustrative uniform-improvement assumption across all serving points: 2/1.2363 = 1.6177x additional throughput, equivalent to 38.185% less time for fixed work. This is not a measured forecast or a universal bound for the scored serving frontier.'))
sanitizer_checks=[]
for job in (report['post_state'] or {}).get('jobs',[]):
    if not job['name'].startswith('sanitizer_'):continue
    log_path=O/(job['name']+'.log')
    log=log_path.read_text(errors='replace') if log_path.exists() else ''
    name=job['name'];mode=name.split('_')[1]
    zero_summary=('RACECHECK SUMMARY: 0 hazards displayed' in log) if mode=='racecheck' else ('ERROR SUMMARY: 0 errors' in log)
    seed=job['command'][-1];rows=job['command'][-3]
    fixture=read(O/'sanitizer'/f'adapter_probe_{seed}_{rows}.json')
    fixture_pass=bool(fixture and fixture.get('complete') and fixture.get('correct') and all(c.get('enabled') and c.get('changed_graph_exact') for c in fixture['cases']))
    sanitizer_checks.append(dict(name=name,returncode=job.get('returncode'),timeout=job.get('timeout',False),zero_error_summary=zero_summary,fixture_pass=fixture_pass,passed=job.get('returncode')==0 and zero_summary and fixture_pass,log=str(log_path)))
report['sanitizer_checks']=sanitizer_checks
report['baseline_unchanged']=report['baseline_sha256']=='8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b'
frozen=read(O/'baseline_delta.json')
report['candidate_hashes_unchanged']=all(patches[n]==frozen[n]['sha256'] for n in patches)
(O/'LOCAL_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
lines=['LOCAL GPU VALIDATION — '+report['recorded_utc'],report['conclusion'],'',f'Matrix complete: {matrix["complete"]}; passed records: {sum(r["passed"] for r in records)}/{len(records)}','']
for r in records:
    status='PASS' if r['passed'] else 'TIMEOUT' if r.get('timeout') else 'FAILED' if r.get('returncode',0)!=0 else 'RUNNING / INCOMPLETE'
    lines.append(f'{status}: {r["name"]}')
lines+=['','Factory state: '+json.dumps(factory),'','Limitations:']+report['limitations']
lines+=['','Official checks:']+[f'{i}: {d.get("status") if d else "unknown"}' for i,d in jobs.items()]
lines+=['',report['old_check_interpretation'],'','Composition state: '+json.dumps(report['composition_state'])]
lines+=['','Large-shape completion runs (four measured rounds plus graph replay):']
for seed,data in report['large_shape_completions'].items():
    passed=bool(data and data.get('complete') and data.get('correct') and all(c.get('chosen') is not None and c.get('changed_graph_backing',{}).get('mismatches')==0 for c in data['cases']))
    lines.append(str(seed)+': '+('PASS' if passed else 'PENDING / INCOMPLETE'))
lines+=['','Post-validation state: '+json.dumps(report['post_state']), '', 'Runtime coverage: raw/live rows must match a registered projection shape. Graph padding does not establish fast-path coverage. Actual serving dispatch frequencies remain unmeasured.', '', 'Performance promotion supported: False', 'Baseline unchanged: '+str(report['baseline_unchanged']), 'Candidate hashes unchanged: '+str(report['candidate_hashes_unchanged'])]
lines+=['','Sanitizer checks (small synthetic geometry):']+[c['name']+': '+('PASS' if c['passed'] else 'FAILED / INCOMPLETE') for c in sanitizer_checks]
lines+=['','Baseline SHA256: '+report['baseline_sha256'],'','No scored submission made by this validation.']
(O/'LOCAL_REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'matrix_complete':matrix['complete'],'passed':sum(r['passed'] for r in records),'started':len(records),'factory':factory}))
