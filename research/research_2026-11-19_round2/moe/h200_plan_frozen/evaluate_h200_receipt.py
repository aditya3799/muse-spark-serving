"""Disabled H200 experiment contract and conservative evidence evaluator.

No kernels, subprocesses, network, GPU API or submissions are invoked. This checks
receipt consistency, not hardware authenticity. Human review remains mandatory.
"""
import hashlib,json,math,statistics
from pathlib import Path

PARENT='8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b'
SGLANG='297e54fd105e974cdf401e5b4d20253a22c16f90'
VARIANTS=('unchanged_lut','finite_only','split_ready','dependent_hint')
SOURCES={
 'unchanged_lut':'7d41368ff71dac0567666f28d7ffffe91fd46d81779ca5c2c803d329515f05f6',
 'finite_only':'3a958850aa6cc360b1c90fe7ba4b36808c57226a6a8b62b4edb7513198afc40e',
 'split_ready':'19ebc0668ad3924f6c26c1cf8465b146a976e98eebdb30675ceefc240c24510b',
 'dependent_hint':'a6394a53e9db8c44d320fde2e74b64a4ede6852c55e4502408d178be138efcc8'}
SHAPES=(8,32,128,256,384,385,753,4097)
ROUTES=('uniform','skew','all_local','all_remote')
LOADS=(1,4,16,32)
MODES=('cold','cache_hit')
LIMIT=2_000_000_000

def sha(value):
 return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def canonical_hash(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def number(value):
 return type(value) in (int,float) and math.isfinite(value)

def check_samples(run):
 """Do not infer a serving score from these component measurements."""
 samples=run.get('times',{})
 names=('humming_parent','candidate','humming_control')
 if set(samples)!=set(names):return False,None
 if not all(isinstance(samples[n],list) for n in names):return False,None
 lengths={len(samples[n]) for n in names if isinstance(samples[n],list)}
 if len(lengths)!=1 or min(lengths)<8 or min(lengths)%2:return False,None
 if not all(number(x) and x>0 for n in names for x in samples[n]):return False,None
 orders=run.get('orders',[])
 if len(orders)!=min(lengths) or orders!=['AB','BA']*(min(lengths)//2):return False,None
 ratios=[p/c for p,c in zip(samples['humming_parent'],samples['humming_control'])]
 controls=all(.95<=x<=1.05 for x in ratios)
 gain=min(statistics.median(samples['humming_parent']),statistics.median(samples['humming_control']))/statistics.median(samples['candidate'])
 return controls,gain

def evaluate(receipt):
 failures=[];diagnostics={}
 def require(condition,name):
  if condition is not True:failures.append(name)
 require(receipt.get('schema_version')==1,'schema_version')
 require(receipt.get('actual_h200_execution') is True,'actual_h200_execution')
 require(receipt.get('baseline_sha256')==PARENT,'frozen_humming_parent')
 require(receipt.get('sglang_commit')==SGLANG,'pinned_sglang')
 require(receipt.get('precision')=='original_nvfp4_w4a16_bf16_fp32_accum','unchanged_precision')
 require(receipt.get('target_hardware')==dict(model='H200',sm=90,tp=8),'h200_tp8')
 require(receipt.get('real_checkpoint_weights') is True,'real_checkpoint_weights')
 require(sha(receipt.get('checkpoint_manifest_sha256')),'checkpoint_manifest')
 provenance=receipt.get('route_holdout',{})
 for field in ('discovery_sha256','holdout_sha256'):
  require(sha(provenance.get(field)),'routes_'+field)
 require(provenance.get('discovery_sha256')!=provenance.get('holdout_sha256'),'distinct_real_route_holdout')
 require(provenance.get('origin')=='captured_real_serving_routes','real_routes_not_graph_dummy')
 require(set(provenance.get('ranks',[]))==set(range(8)),'all_rank_routes')
 require(len(set(provenance.get('layers',[])))>=3,'multi_layer_routes')
 require(provenance.get('heldout_before_selection') is True,'predeclared_route_holdout')
 coefficient=receipt.get('coefficients',{})
 require(coefficient.get('finite_pairs_exact')==4064,'all_finite_coefficient_pairs')
 require(coefficient.get('nan_pairs_rejected')==32,'nan_scale_fallback')
 for field in ('signed_zero_exact','subnormal_exact','sm90_executed','lut_fallback_exact'):
  require(coefficient.get(field) is True,'coefficient_'+field)
 required={(m,r) for m in SHAPES for r in ROUTES}
 variants=receipt.get('variants',{})
 require(set(variants)==set(VARIANTS),'four_separate_variants')
 for name in VARIANTS:
  v=variants.get(name,{})
  require(v.get('source_sha256')==SOURCES[name],name+'_source_binding')
  require(v.get('complete') is True,name+'_complete')
  require(v.get('activation_logged') is True,name+'_dispatch')
  require(v.get('raw_live_exact_vs_lut') is True,name+'_raw_math')
  require(v.get('final_all_bytes_exact_vs_lut') is True,name+'_final_math')
  require(v.get('humming_comparison_scope')=='both_gemms_conversion_routing_finalize',name+'_whole_ffn_parent_scope')
  require(v.get('parent_precision_audit') is True,name+'_parent_precision_audit')
  # Ordinary FP32 reordering vs Humming is characterized; it is not itself a precision violation.
  for field in ('max_abs_error_vs_humming','rms_error_vs_humming'):
   require(number(v.get(field)) and v.get(field)>=0,name+'_'+field)
  rows=v.get('whole_ffn',[])
  keys=[(r.get('m'),r.get('route')) for r in rows]
  require(len(keys)==len(set(keys)) and set(keys)==required,name+'_shape_route_coverage')
  summary=[]
  for row in rows:
   runs=row.get('runs',[]);seeds=[r.get('seed') for r in runs]
   okay=len(runs)==2 and all(type(s) is int for s in seeds) and len(set(seeds))==2
   measured=[]
   for run in runs:
    controls,gain=check_samples(run);measured.append(gain)
    okay=okay and controls and run.get('complete') is True and run.get('correct') is True
   require(okay,f"{name}_controlled_{row.get('m')}_{row.get('route')}")
   summary.append(dict(m=row.get('m'),route=row.get('route'),conservative_median_ratios=measured))
  diagnostics[name]=summary
 graph=receipt.get('graph_liveness',{})
 for field in ('warmup_capture_and_eager','changed_inputs_and_routes','mixed_shape_sequence',
               'generation_wraparound','phase_a_b_progress','watchdog_complete','owner_retire_reload',
               'unknown_stream_fallback','nan_scale_fallback','all_variants_executed'):
  require(graph.get(field) is True,'graph_'+field)
 require(type(graph.get('replays')) is int and graph.get('replays',0)>=256,'long_graph_replay')
 owner=receipt.get('owner_contract',{})
 for field in ('canonical_lease','scan_identity_atomic','stages_weakly_bound','reentrant_retire_rejected',
               'all_writes_retire_first','all_graphs_retired_before_mutation','all_referencing_streams_synchronized'):
  require(owner.get(field) is True,'owner_'+field)
 memory=receipt.get('memory',{})
 require(memory.get('mem_fraction_static')==.85,'fixed_memory_fraction')
 require(memory.get('global_peak_counters_reset') is False,'no_production_peak_reset')
 require(memory.get('scope')=='full_model_startup_capture_and_serving','full_model_memory_scope')
 for phase in ('startup','capture','serving'):
  d=memory.get(phase,{})
  for metric in ('allocated','reserved'):
   parent=d.get('parent_peak_'+metric);candidate=d.get('candidate_peak_'+metric)
   valid=number(parent) and number(candidate) and parent>=0 and candidate>=0
   require(valid and candidate-parent<=LIMIT,phase+'_'+metric+'_extra_limit')
 require(memory.get('measured_with_all_enabled_features') is True,'joint_feature_memory')
 serving=receipt.get('serving',{})
 require(sha(serving.get('integrated_patch_sha256')),'integrated_patch_binding')
 manifest=serving.get('measured_source_manifest',{})
 require(manifest.get('patch_sha256')==serving.get('integrated_patch_sha256'),'measured_patch_binding')
 require(manifest.get('baseline_sha256')==PARENT,'measured_baseline_binding')
 require(manifest.get('kernel_sources')==SOURCES,'measured_kernel_sources')
 require(manifest.get('checkpoint_manifest_sha256')==receipt.get('checkpoint_manifest_sha256'),'measured_checkpoint_binding')
 require(manifest.get('route_holdout_sha256')==provenance.get('holdout_sha256'),'measured_route_binding')
 policy=serving.get('selected_policy',{})
 require(set(policy)=={'decode','prefill','fallback'},'selected_policy_scope')
 require(all(v in set(VARIANTS)|{'humming_parent'} for v in policy.values()),'selected_policy_variants')
 require(manifest.get('selected_policy')==policy,'measured_policy_binding')
 require(serving.get('measured_source_manifest_sha256')==canonical_hash(manifest),'source_manifest_hash')

 require(serving.get('all_kernel_variants_not_jointly_enabled') is True,'single_selected_kernel_policy')
 require(serving.get('all_routes_served') is True,'no_route_dropping')
 require(serving.get('judge_output_gate')=='PASS','official_output_quality')
 rows=serving.get('scenarios',[]);keys=[(r.get('concurrency'),r.get('mode')) for r in rows]
 require(len(keys)==len(set(keys)) and set(keys)=={(c,m) for c in LOADS for m in MODES},'full_serving_frontier')
 for row in rows:
  prefix=f"serving_{row.get('concurrency')}_{row.get('mode')}_"
  for field in ('complete','matched_prompts','matched_cache','paired_parent','independent_holdout','dispatch_logged','token_gate_passed','ttft_gate_passed','tpot_gate_passed'):
   require(row.get(field) is True,prefix+field)
  require(sha(row.get('receipt_sha256')),prefix+'receipt')
  require(row.get('source_manifest_sha256')==serving.get('measured_source_manifest_sha256'),prefix+'measured_source_binding')
  for metric in ('throughput_ratio','ttft_ratio','tpot_ratio'):
   require(number(row.get(metric)) and row.get(metric)>0,prefix+metric)
 frontier=serving.get('frontier_score',{})
 require(frontier.get('source')=='official_full_frontier_methodology','official_scoring_scope')
 require(frontier.get('all_concurrencies_complete') is True,'complete_official_frontier')
 require(frontier.get('not_projected_from_components') is True,'no_component_projection')
 require(sha(frontier.get('methodology_sha256')),'methodology_binding')
 estimate=frontier.get('ratio_over_stock');lower=frontier.get('independent_lower_bound_over_stock')
 require(number(estimate) and number(lower) and estimate>=lower>=2.,'conclusive_independent_2x_serving')
 return dict(schema_version=1,decision='READY_FOR_HUMAN_REVIEW' if not failures else 'HOLD',
             failing_gates=failures,component_diagnostics=diagnostics,
             reported_gates_consistent=not failures,conclusive_2x_evidence=False,
             independent_frontier_recomputation_complete=False,scored_submission_authorized=False,
             evaluator_scope='Reported-receipt consistency only. Official frontier scalars are not independently recomputed or authenticated here; conclusive_2x_evidence is always false. Human review must verify full raw official results. No component rewards are summed or multiplied.')

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('receipt',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
 raw=a.receipt.read_bytes();r=evaluate(json.loads(raw.decode('utf-8-sig')));r['input_sha256']=hashlib.sha256(raw).hexdigest()
 if a.output:a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in r.items() if k!='component_diagnostics'},indent=2,allow_nan=False))

