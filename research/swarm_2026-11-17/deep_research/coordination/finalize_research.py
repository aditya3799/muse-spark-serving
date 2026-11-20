import argparse,datetime,hashlib,json,pathlib,math
ROOT=pathlib.Path(__file__).resolve().parents[3];P=pathlib.Path(__file__).resolve().parent;D=P.parent
p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
state=read(P/'SESSION_STATE.json');now=datetime.datetime.now(datetime.timezone.utc);start=datetime.datetime.fromisoformat(state['start_utc'][:26]+'+00:00');elapsed=(now-start).total_seconds()
if a.final:assert elapsed>=7200,elapsed
baseline=[]
for item in state['baseline_files']:
 path=ROOT/item['path'];actual=sha(path);assert actual==item['sha256'];baseline.append({'path':item['path'],'sha256':actual,'unchanged':True})
reports=[P/'W13_FUSION_RESEARCH.json',P/'W13_NATIVE_E32_REVIEW.json',P/'SELECTOR_PACK_MANIFEST.json',P/'DRAFT_FC_SHARDING_FINDINGS.json',P/'FA3_POLICY_AUDIT.json',P/'SYSTEMS_BASELINE_REVIEW.json',P/'SM90_EXACT_POLICY_PAIRINGS.json',P/'DFLASH_SYSTEMS_SOURCE_AUDIT.json',P/'remote_6570fbd9e1b64ce4_refresh_02.json',D/'expert_partition/LANE_CHECKPOINT.json',D/'expert_partition/EVIDENCE_SUMMARY.json',D/'expert_partition/SANITIZER_COMPLETENESS_REVIEW.json',D/'expert_partition/ROOT_W13_NATIVE_E32_CPU_REVIEW.json',D/'expert_partition/ROOT_W13_NATIVE_E32_FIXED_CPU_REVIEW.json',D/'hopper_moe/DEEP_RESEARCH_CHECKPOINT.json',D/'hopper_moe/H200_VIRTUAL_K_GATE_PLAN.json',D/'hopper_moe/H200_W13_FUSED_GATE_PLAN.json',D/'hopper_moe/W13_FUSED_SM90_SASS_AUDIT.json',D/'aux_speculation/AUX_RESEARCH_REPORT.json',D/'aux_speculation/decode_extension/DECODE_RESEARCH_REPORT.json',D/'aux_speculation/decode_extension/INTEGRATION_RISK_REPORT.json']
assert all(x.exists() for x in reports),[str(x) for x in reports if not x.exists()]
indexed=[{'path':str(x.relative_to(ROOT)),'sha256':sha(x),'bytes':x.stat().st_size} for x in reports]
receipts=[]
for folder in sorted(P.iterdir()):
 if not folder.is_dir():continue
 receipt=folder/'run_receipt.json'
 if not receipt.exists():continue
 r=read(receipt);row={'path':str(receipt.relative_to(ROOT)),'sha256':sha(receipt),'status':r.get('status'),'exit_code':r.get('exit_code')}
 snapshot=folder/'source_snapshot.py'
 if snapshot.exists():row['source_snapshot_sha256']=sha(snapshot);row['snapshot_matches_receipt']=sha(snapshot)==r.get('source_sha256');assert row['snapshot_matches_receipt']
 result=folder/'results.json'
 if result.exists():row['result_sha256']=sha(result);row['result_status']=read(result).get('status')
 receipts.append(row)
checks={}
for name in ('w13_fusion_gpu_02','w13_fusion_chain_01','w13_fusion_native_e32_01','w13_fusion_finite_stress_01','w13_fusion_empty_rank_01'):
 r=read(P/name/'results.json');receipt=read(P/name/'run_receipt.json');assert receipt['exit_code']==0
 assert all(x['bitwise'] and math.isfinite(x.get('max_abs',0)) for x in r['checks'])
 if name in ('w13_fusion_finite_stress_01','w13_fusion_empty_rank_01'):assert all(x['finite'] for x in r['checks'])
 checks[name]={'bitwise_checks':len(r['checks']),'scope':r['scope'],'result_sha256':sha(P/name/'results.json')}
summary={'status':'RESEARCH_PASS_SAVED_TARGET_NOT_REACHED' if a.final else 'DRAFT','started_utc':start.isoformat(),'finished_utc':now.isoformat(),'elapsed_seconds':elapsed,'elapsed_hours':elapsed/3600,'agents_total':4,'new_submissions':0,'new_remote_checks':0,'paid_gpu_rentals':0,'confirmed_serving_score':1.2237,'target_screenshot_score':3.813,'baseline':baseline,'w13_checks':checks,'principal_limits':['No H200 numerical or serving execution','Full-helper shared-memory racecheck incomplete; do not claim clean','Original decode owner variants held, corrected hot fused has graph evidence','W13 component timing modest and local; scratch savings structural, integration pending','No multiplicative extrapolation from component ratios to serving score'],'indexed_reports':indexed,'root_run_receipts':receipts}
(P/'EXPERIMENT_INDEX.json').write_text(json.dumps(summary,indent=2)+'\n')
if a.final:
 state.update(status='RESEARCH_CHECKPOINT_SAVED_TARGET_NOT_REACHED',gpu_compiler_owner=None,last_updated_utc=now.isoformat(),elapsed_hours_at_checkpoint=elapsed/3600,finished_utc=now.isoformat());(P/'SESSION_STATE.json').write_text(json.dumps(state,indent=2)+'\n')
 handoff=P/'RESEARCH_HANDOFF.txt';s=handoff.read_text();s=s.replace('Status: IN PROGRESS. Final verification and elapsed-time receipt pending.', 'Status: Research checkpoint saved; serving target not reached.\nDuration: '+str(round(elapsed/3600,3))+' hours, '+start.isoformat()+' to '+now.isoformat()+'.\nNo background GPU work remains; no automatic continuation is scheduled.')
 s=s.replace('1. Finish the targeted finite/subnormal stress receipt and freeze source hashes; check EXPERIMENT_INDEX.json for final status.', '1. Read EXPERIMENT_INDEX.json. Finite/subnormal stress passes84 checks; empty-rank boundary passes42. All root run receipts and key lane reports are hashed. No new runs were submitted.')
 handoff.write_text(s)
 previous=ROOT/'SESSION_HANDOFF.txt';backup=P/'SESSION_HANDOFF.before_research.txt'
 if not backup.exists():backup.write_bytes(previous.read_bytes())
 prefix='LATEST RESEARCH CHECKPOINT — 2026-11-17\n\nRead swarm_2026-11-17/deep_research/coordination/RESEARCH_HANDOFF.txt first.\nEvidence: swarm_2026-11-17/deep_research/coordination/EXPERIMENT_INDEX.json\nFour-agent research duration: '+str(round(elapsed/3600,3))+' hours.\nConfirmed serving score remains1.2237x; screenshot target3.813x not reached.\nNo new submissions or remote checks. Scored candidate and root patch unchanged.\nPrevious handoff is preserved verbatim below and in the research backup.\n\n--- PREVIOUS HANDOFF ---\n\n'
 previous.write_text(prefix+backup.read_text(encoding='utf-8-sig'),encoding='utf-8')
 print(json.dumps({'status':summary['status'],'elapsed_hours':elapsed/3600,'reports':len(indexed),'root_receipts':len(receipts),'baseline_unchanged':True,'w13_checks':{k:v['bitwise_checks'] for k,v in checks.items()}}))
else:print(json.dumps({'status':'DRAFT','elapsed_hours':elapsed/3600,'reports':len(indexed),'root_receipts':len(receipts)}))
