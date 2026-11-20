"""Synthetic contract tests only; this file contains no H200 experiment evidence."""
import copy,json,math
from pathlib import Path
from evaluate_h200_receipt import *

def valid():
 r=dict(schema_version=1,actual_h200_execution=True,baseline_sha256=PARENT,sglang_commit=SGLANG,
  precision='original_nvfp4_w4a16_bf16_fp32_accum',target_hardware=dict(model='H200',sm=90,tp=8),
  real_checkpoint_weights=True,checkpoint_manifest_sha256='1'*64,
  route_holdout=dict(discovery_sha256='2'*64,holdout_sha256='3'*64,origin='captured_real_serving_routes',ranks=list(range(8)),layers=[1,17,40],heldout_before_selection=True),
  coefficients=dict(finite_pairs_exact=4064,nan_pairs_rejected=32,signed_zero_exact=True,subnormal_exact=True,sm90_executed=True,lut_fallback_exact=True),variants={},
  graph_liveness={k:True for k in ('warmup_capture_and_eager','changed_inputs_and_routes','mixed_shape_sequence','generation_wraparound','phase_a_b_progress','watchdog_complete','owner_retire_reload','unknown_stream_fallback','nan_scale_fallback','all_variants_executed')},
  owner_contract={k:True for k in ('canonical_lease','scan_identity_atomic','stages_weakly_bound','reentrant_retire_rejected','all_writes_retire_first','all_graphs_retired_before_mutation','all_referencing_streams_synchronized')},
  memory=dict(mem_fraction_static=.85,global_peak_counters_reset=False,scope='full_model_startup_capture_and_serving',measured_with_all_enabled_features=True),
  serving=dict(integrated_patch_sha256='4'*64,all_kernel_variants_not_jointly_enabled=True,all_routes_served=True,judge_output_gate='PASS',scenarios=[],frontier_score=dict(source='official_full_frontier_methodology',all_concurrencies_complete=True,not_projected_from_components=True,methodology_sha256='5'*64,ratio_over_stock=2.2,independent_lower_bound_over_stock=2.05)))
 r['graph_liveness']['replays']=256
 for phase in ('startup','capture','serving'):
  r['memory'][phase]=dict(parent_peak_allocated=100_000_000_000,candidate_peak_allocated=101_000_000_000,parent_peak_reserved=101_000_000_000,candidate_peak_reserved=102_000_000_000)
 for name in VARIANTS:
  v=dict(source_sha256=SOURCES[name],complete=True,activation_logged=True,raw_live_exact_vs_lut=True,final_all_bytes_exact_vs_lut=True,humming_comparison_scope='both_gemms_conversion_routing_finalize',parent_precision_audit=True,max_abs_error_vs_humming=.002,rms_error_vs_humming=.0001,whole_ffn=[])
  for m in SHAPES:
   for route in ROUTES:
    runs=[]
    for seed in (791,997):
     # Old LUT may regress: diagnostic baselines never need to win to be measured.
     candidate=200. if name=='unchanged_lut' else 50.
     runs.append(dict(seed=seed,complete=True,correct=True,orders=['AB','BA']*4,times=dict(humming_parent=[100.]*8,candidate=[candidate]*8,humming_control=[100.]*8)))
    v['whole_ffn'].append(dict(m=m,route=route,runs=runs))
  r['variants'][name]=v
 for c in LOADS:
  for mode in MODES:
   row={k:True for k in ('complete','matched_prompts','matched_cache','paired_parent','independent_holdout','dispatch_logged','token_gate_passed','ttft_gate_passed','tpot_gate_passed')}
   row.update(concurrency=c,mode=mode,receipt_sha256='6'*64,throughput_ratio=1.8,ttft_ratio=.6,tpot_ratio=.6)
   r['serving']['scenarios'].append(row)
 policy=dict(decode='finite_only',prefill='humming_parent',fallback='unchanged_lut')
 manifest=dict(patch_sha256='4'*64,baseline_sha256=PARENT,kernel_sources=SOURCES,checkpoint_manifest_sha256='1'*64,route_holdout_sha256='3'*64,selected_policy=policy)
 r['serving'].update(measured_source_manifest=manifest,measured_source_manifest_sha256=canonical_hash(manifest),selected_policy=policy)
 for row in r['serving']['scenarios']:row['source_manifest_sha256']=canonical_hash(manifest)
 return r

base=valid();result=evaluate(base)
assert not result['failing_gates'],result['failing_gates']
assert result['decision']=='READY_FOR_HUMAN_REVIEW' and result['scored_submission_authorized'] is False
assert result['reported_gates_consistent'] and not result['conclusive_2x_evidence']
passed=[]
def reject(label,edit):
 r=copy.deepcopy(base);edit(r)
 out=evaluate(r)
 assert out['decision']=='HOLD' and not out['conclusive_2x_evidence'] and not out['scored_submission_authorized'],label
 passed.append(dict(case=label,veto=out['failing_gates']))

reject('no_receipt',lambda r:r.clear())
reject('unbound_integrated_patch',lambda r:r['serving'].update(integrated_patch_sha256='a'*64))
reject('unbound_selected_policy',lambda r:r['serving']['selected_policy'].update(decode='dependent_hint'))
reject('unbound_scenario_manifest',lambda r:r['serving']['scenarios'][0].update(source_manifest_sha256='b'*64))
reject('unbound_measured_checkpoint',lambda r:r['serving']['measured_source_manifest'].update(checkpoint_manifest_sha256='c'*64))
reject('wrong_hardware',lambda r:r['target_hardware'].update(sm=86))
reject('component_only_projection',lambda r:r['serving']['frontier_score'].update(not_projected_from_components=False))
reject('point_above2_lower_below2',lambda r:r['serving']['frontier_score'].update(independent_lower_bound_over_stock=1.99))
reject('C16_only',lambda r:r['serving'].update(scenarios=[x for x in r['serving']['scenarios'] if x['concurrency']==16]))
reject('no_cold',lambda r:r['serving'].update(scenarios=[x for x in r['serving']['scenarios'] if x['mode']!='cold']))
reject('fake_graph_routes',lambda r:r['route_holdout'].update(origin='cuda_graph_dummy'))
reject('same_route_holdout',lambda r:r['route_holdout'].update(holdout_sha256=r['route_holdout']['discovery_sha256']))
reject('missing_rank',lambda r:r['route_holdout'].update(ranks=[0]))
reject('resident_only_memory',lambda r:r['memory'].update(scope='single_layer_owned_tensors'))
reject('startup_budget_one_byte_over',lambda r:r['memory']['startup'].update(candidate_peak_allocated=r['memory']['startup']['parent_peak_allocated']+LIMIT+1))
reject('reserved_budget_one_byte_over',lambda r:r['memory']['capture'].update(candidate_peak_reserved=r['memory']['capture']['parent_peak_reserved']+LIMIT+1))
reject('reset_global_peak',lambda r:r['memory'].update(global_peak_counters_reset=True))
reject('changed_precision',lambda r:r.update(precision='nvfp4_activations'))
reject('wrong_parent',lambda r:r.update(baseline_sha256='0'*64))
reject('nan_error_metric',lambda r:r['variants']['finite_only'].update(max_abs_error_vs_humming=float('nan')))
reject('changed_source',lambda r:r['variants']['dependent_hint'].update(source_sha256='0'*64))
reject('missing_boundary385',lambda r:r['variants']['finite_only'].update(whole_ffn=[x for x in r['variants']['finite_only']['whole_ffn'] if x['m']!=385]))
reject('raw_exact_final_wrong',lambda r:r['variants']['split_ready'].update(final_all_bytes_exact_vs_lut=False))
reject('same_seed',lambda r:r['variants']['finite_only']['whole_ffn'][0]['runs'][1].update(seed=791))
reject('missing_timing_control',lambda r:r['variants']['finite_only']['whole_ffn'][0]['runs'][0]['times'].pop('humming_control'))
reject('malformed_timing_control',lambda r:r['variants']['finite_only']['whole_ffn'][0]['runs'][0]['times'].update(humming_control='invalid'))
reject('infinite_component_speed',lambda r:r['variants']['finite_only']['whole_ffn'][0]['runs'][0]['times'].update(candidate=[float('inf')]*8))
reject('noisy_parent_despite100x_gain',lambda r:r['variants']['finite_only']['whole_ffn'][0]['runs'][0]['times'].update(candidate=[1.]*8,humming_control=[80.]*8))
reject('unbalanced_order',lambda r:r['variants']['finite_only']['whole_ffn'][0]['runs'][0].update(orders=['AB']*8))
for section in ('coefficients','graph_liveness','owner_contract'):
 for field,value in base[section].items():
  if value is True:reject(section+'_'+field,lambda r,s=section,f=field:r[s].update({f:False}))
for index,row in enumerate(base['serving']['scenarios']):
 for field in ('token_gate_passed','ttft_gate_passed','tpot_gate_passed'):
  reject('serving_veto_'+str(index)+'_'+field,lambda r,i=index,f=field:r['serving']['scenarios'][i].update({f:False}))
# No component gain changes global readiness without every independent system gate.
r=valid();r.pop('serving');assert evaluate(r)['decision']=='HOLD';passed.append(dict(case='all_components_win_no_serving',veto='serving absent'))
# Reported scalar claims alone can never establish independently conclusive evidence.
r=valid()
for row in r['serving']['scenarios']:row['throughput_ratio']=.01
assert evaluate(r)['conclusive_2x_evidence'] is False
passed.append(dict(case='unverified_frontier_claim_never_conclusive',veto='independent frontier verification absent'))
out=dict(complete=True,synthetic_contract_tests_only=True,actual_h200_execution=False,positive_consistency_fixture=1,negative_cases=len(passed),all_submit_authorization_false=True,cases=passed)
Path(__file__).with_name('h200_evaluator_contracts.json').write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in out.items() if k!='cases'},indent=2))
