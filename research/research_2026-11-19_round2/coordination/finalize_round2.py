import pathlib,json,hashlib,datetime,sys,subprocess
HERE=pathlib.Path(__file__).resolve().parent
ROUND=HERE.parent;ROOT=ROUND.parent
now=datetime.datetime.now(datetime.timezone.utc)
end=datetime.datetime(2026,11,19,13,32,37,tzinfo=datetime.timezone.utc)
finish='--finish' in sys.argv
if finish and now<end:raise RuntimeError('Requested additional three-hour period is not complete')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected={
 'release_2026-11-18/candidate.diff':'8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b',
 'research_2026-11-19/prefill/integration_joint/candidate.diff':'530e9ccc160398e82386f6e455cdbf80fa4d40f43ec1209b1de7290a5276a8c6',
 'research_2026-11-19/moe/normal_package/candidate.diff':'80db9ce3d958c58e31be4b64f06442d0ef152493221c3927ad9c43deab68319f'}
integrity=[]
checkpoint=json.loads((HERE/'integrity_checkpoint.json').read_text())
for row in checkpoint['modified_files']:
 assert sha(ROOT/'sglang'/row['path'])==row['sha256'], 'Serving source changed since integrity checkpoint'

for rel,want in expected.items():
 p=ROOT/rel;got=sha(p);assert got==want,(rel,got,want)
 integrity.append({'path':rel,'bytes':p.stat().st_size,'sha256':got,'unchanged':True})
paths=[
 'coordination/RESEARCH_DECISION.txt','coordination/TOOLCHAIN_PROVENANCE.json','coordination/RESOURCE_FINISH.json','coordination/FORECAST_LIMITS.json','coordination/forecast_sensitivity.json','coordination/NEXT_EXPERIMENTS.json','coordination/integrity_checkpoint.json','coordination/final_index_integrity.json','coordination/timing_receipt_consistency.json',
 'coordination/target_argmax_postops/correctness.json','coordination/target_argmax_postops/timing_device_guard.json',
 'coordination/softcap_parallel/correctness.json','coordination/softcap_parallel/timing.json','coordination/softcap_parallel/DECISION.json',
 'coordination/finalize_reduce/FINALIZER_EVIDENCE.json','coordination/finalize_reduce/PROTOCOL_READOUT.txt','coordination/finalize_reduce/steady_bench.json','coordination/finalize_reduce/protocol_probe.json',
 'coordination/gather_norm/PROTOCOL_REVIEW.txt','coordination/gather_norm/source_manifest.json',
 'coordination/projection_failure_policy/manifest.json','coordination/projection_failure_policy/composition_validation.json',
 'coordination/compression_audit.json',
 'coordination/donor_audit/flashsampling/ASSESSMENT.txt','coordination/donor_audit/servingstudio/applicability.json',
 'decode/ROUND2_DECODE_READOUT.txt','decode/DECODE_INDEX.json','decode/freeze_recheck.json',
 'decode/target_selector_freeze_manifest.json','decode/sampler_freeze_manifest.json','decode/MUSE_LOGITS_ACTIVATION.txt','decode/FLASHINFER_NOV_READOUT.txt','decode/cluster_softcap/PLAN.txt',
 'moe/pr50800_frozen/manifest.json','moe/pr50800_frozen_v2/manifest.json','moe/ROUND2_MOE_EVIDENCE.json','moe/h200_plan_frozen/manifest.json','moe/PR50800_READOUT.txt','moe/finite_frozen/manifest.json','moe/split_readiness_frozen/manifest.json','moe/h200_current_readiness.json',
 'prefill/ARTIFACT_INDEX.json','prefill/ROUND2_STATUS.txt','prefill/PREFILL_BCG_INTEGRATION_PLAN.txt','prefill/TERMINAL_FFN_REVIEW.txt','prefill/combined_memory_ledger.json','prefill/prologue_warp8/byte_matrix.json']
# Final optional receipts become required when the overall period is closed.
final_paths=['coordination/wgmma_hazard_audit/READOUT.txt','coordination/wgmma_hazard_audit/FINAL_MANIFEST.json']
if finish:paths+=final_paths
else:paths += [s for s in final_paths if (ROUND/s).is_file()]
artifacts=[]
for rel in paths:
 p=ROUND/rel;assert p.is_file(),f'Missing evidence: {rel}'
 artifacts.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha(p)})
remote_history=ROOT/'research_2026-11-19/coordination/check_status_history.jsonl'
latest={}
for line in remote_history.read_text().splitlines():
 row=json.loads(line)
 if row.get('job'):latest[row['job']]=row
assert len(latest)==4
report={'generated_utc':now.isoformat(),'additional_research_period_complete':finish,'official_score':1.2363,'conclusive_2x_serving':False,'scored_submission_recommended':False,'scored_jobs_launched_this_round':0,'new_remote_checks_launched_this_round':0,'protected_payload_integrity':integrity,'latest_remote_statuses':list(latest.values()),'evidence':artifacts,'scope':'Root index binds final readouts and lane indexes. Linked lane indexes bind detailed sources, numerical/timing receipts and rejected controls. Local components are not serving forecasts.'}
(HERE/'EVIDENCE_INDEX.json').write_text(json.dumps(report,indent=2)+'\n')
s=json.loads((HERE/'SESSION_STATE.json').read_text());s['last_checkpoint_utc']=now.isoformat();s['latest_remote_statuses']=list(latest.values());s['evidence_index_sha256']=sha(HERE/'EVIDENCE_INDEX.json');s['protected_payload_integrity']=integrity
s['tracked_source_hashes_match_checkpoint']=True
s['last_remote_status']='; '.join(row['job']+': '+str(row.get('status',row.get('error')))+' at '+row.get('checked_utc','unknown') for row in latest.values())
s['findings']['new_moe_donor']='OpenPR50800 original and BF16-boundary-restored variants compile SM90a. Separate raw-scale/original-code CPU layout is reversible across all256scale bytes; rawsourceuncompiled, GPUconverter/loadermissing. FullFFN and H200 token/quality gatesremain.'
if finish:
 start=datetime.datetime.fromisoformat(s['started_utc'].replace('Z','+00:00'))
 s.update(status='Completed the requested additional three-hour research period; no conclusive2x and no scored submission.',completed_utc=now.isoformat(),elapsed_seconds=(now-start).total_seconds(),scored_jobs_launched_this_round=0,new_remote_checks_launched_this_round=0)
(HERE/'SESSION_STATE.json').write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps({'complete':finish,'artifacts':len(artifacts),'protected_payloads':len(integrity),'remote':[{k:v for k,v in row.items() if k in ('job','status','checked_utc','error')} for row in latest.values()],'elapsed_seconds':s.get('elapsed_seconds')},indent=2))
