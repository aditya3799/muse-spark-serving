"""Emit an intentionally failing blank schema, never synthetic positive evidence."""
from pathlib import Path
import json
from evaluate_h200_receipt import *
r=dict(schema_version=1,template_only=True,actual_h200_execution=False,baseline_sha256=PARENT,sglang_commit=SGLANG,precision='original_nvfp4_w4a16_bf16_fp32_accum',target_hardware=dict(model='H200',sm=90,tp=8),real_checkpoint_weights=False,checkpoint_manifest_sha256=None,
 route_holdout=dict(discovery_sha256=None,holdout_sha256=None,origin=None,ranks=[],layers=[],heldout_before_selection=False),
 coefficients=dict(finite_pairs_exact=0,nan_pairs_rejected=0,signed_zero_exact=False,subnormal_exact=False,sm90_executed=False,lut_fallback_exact=False),variants={},
 graph_liveness={k:False for k in ('warmup_capture_and_eager','changed_inputs_and_routes','mixed_shape_sequence','generation_wraparound','phase_a_b_progress','watchdog_complete','owner_retire_reload','unknown_stream_fallback','nan_scale_fallback','all_variants_executed')},
 owner_contract={k:False for k in ('canonical_lease','scan_identity_atomic','stages_weakly_bound','reentrant_retire_rejected','all_writes_retire_first','all_graphs_retired_before_mutation','all_referencing_streams_synchronized')},
 memory=dict(mem_fraction_static=.85,global_peak_counters_reset=False,scope='full_model_startup_capture_and_serving',measured_with_all_enabled_features=False),
 serving=dict(integrated_patch_sha256=None,measured_source_manifest={},measured_source_manifest_sha256=None,selected_policy={},all_kernel_variants_not_jointly_enabled=False,all_routes_served=False,judge_output_gate=None,scenarios=[],frontier_score=dict(source=None,all_concurrencies_complete=False,not_projected_from_components=False,methodology_sha256=None,ratio_over_stock=None,independent_lower_bound_over_stock=None)))
r['graph_liveness']['replays']=0
for phase in ('startup','capture','serving'):
 r['memory'][phase]={k:None for k in ('parent_peak_allocated','candidate_peak_allocated','parent_peak_reserved','candidate_peak_reserved')}
for name in VARIANTS:
 r['variants'][name]=dict(source_sha256=SOURCES[name],complete=False,activation_logged=False,raw_live_exact_vs_lut=False,final_all_bytes_exact_vs_lut=False,humming_comparison_scope=None,parent_precision_audit=False,max_abs_error_vs_humming=None,rms_error_vs_humming=None,whole_ffn=[dict(m=m,route=route,runs=[]) for m in SHAPES for route in ROUTES])
for c in LOADS:
 for mode in MODES:
  row={k:False for k in ('complete','matched_prompts','matched_cache','paired_parent','independent_holdout','dispatch_logged','token_gate_passed','ttft_gate_passed','tpot_gate_passed')}
  row.update(concurrency=c,mode=mode,receipt_sha256=None,source_manifest_sha256=None,throughput_ratio=None,ttft_ratio=None,tpot_ratio=None)
  r['serving']['scenarios'].append(row)
assert evaluate(r)['decision']=='HOLD'
Path(__file__).with_name('h200_receipt_template.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
print('Blank schema saved; all execution and performance gates unfulfilled')
