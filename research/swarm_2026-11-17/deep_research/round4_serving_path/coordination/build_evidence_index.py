import datetime,hashlib,json
from pathlib import Path
C=Path(__file__).resolve().parent;P=C.parent;R=P.parents[2]
items=[
 ('organizer_supplied_decisions','coordination/ORGANIZER_CLARIFICATION.json',False),
 ('new_candidate_criteria','coordination/REVISED_RESEARCH_DECISION_CRITERIA.json',False),
 ('organizer_expert_tp_source','expert_partition/ORGANIZER_CLARIFICATION_EXPERT_TP.json',False),
 ('organizer_swa_source','aux_speculation/ORGANIZER_CLARIFICATION_SWA.json',False),
 ('organizer_reduction_source','hopper_moe/ORGANIZER_CLARIFICATION_REDUCTION.json',False),
 ('organizer_drafting_source','aux_speculation/ORGANIZER_CLARIFICATION_DRAFTING.json',False),
 ('organizer_fa3_source','hopper_moe/ORGANIZER_CLARIFICATION_FA3.json',False),
 ('organizer_liveness_source','expert_partition/ORGANIZER_CLARIFICATION_LIVENESS.json',False),
 ('session_final','coordination/SESSION_STATE.json',False),
 ('saved_serving_result','swarm_2026-11-16/coordination/status_e399c79bb1994630.json',True),
 ('saved_scored_patch','swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate.diff',True),
 ('preservation','coordination/FINAL_PRESERVATION_AUDIT.json',False),
 ('summary','coordination/RESEARCH_FINDINGS.json',False),
 ('handoff','coordination/RESEARCH_HANDOFF.txt',False),
 ('diagnostics_documentation','coordination/PROBE_README.txt',False),
 ('local_cuda_metadata','aux_speculation/metadata_gpu_02/probe/results.json',False),
 ('local_cuda_metadata_receipt','aux_speculation/metadata_gpu_02/run_receipt.json',False),
 ('local_cuda_fullshape_aux','aux_speculation/fullshape_aux_gpu_02/probe/results.json',False),
 ('local_cuda_fullshape_aux_receipt','aux_speculation/fullshape_aux_gpu_02/run_receipt.json',False),
 ('default_off_prefix_overlay','aux_speculation/graph_prefix_overlay.diff',False),
 ('prefix_qualification','aux_speculation/QUALIFICATION_LEDGER.json',False),
 ('segmented_source_feasibility','aux_speculation/SEGMENTED_VERIFY_FEASIBILITY.json',False),
 ('segmented_accept_buffer_negative','aux_speculation/SEGMENTED_ACCEPT_BUFFER_CPU.json',False),
 ('segmented_aux_cpu','aux_speculation/SEGMENTED_AUX_STAGING_CPU.json',False),
 ('future_memory_source_only','aux_speculation/FUTURE_MEMORY_OPTION.json',False),
 ('source_metric_audit','expert_partition/METRIC_AND_WORKLOAD_AUDIT.json',False),
 ('source_scheduler_audit','expert_partition/SCHEDULER_POLICY_FEASIBILITY.json',False),
 ('source_collective_audit','expert_partition/TARGET_ARGMAX_COLLECTIVE_AND_BUCKET_AUDIT.json',False),
 ('argmax_qualification','expert_partition/TARGET_ARGMAX_QUALIFICATION.json',False),
 ('argmax_padded_manifest','expert_partition/TARGET_ARGMAX_PADDED_V2_MANIFEST.json',False),
 ('argmax_padded_incremental_patch','expert_partition/target_argmax_padded_v2_incremental.diff',False),
 ('local_cuda_argmax_lifecycle','coordination/target_argmax_overlay_01/probe/results.json',False),
 ('local_cuda_argmax_padded','coordination/target_argmax_padded_01/probe/results.json',False),
 ('local_cuda_argmax_padded_receipt','coordination/target_argmax_padded_01/run_receipt.json',False),
 ('local_cuda_recorder_prior_fixture','coordination/phase_probe_04/results.json',False),
 ('local_cuda_recorder_prime_delta','coordination/phase_prime_stream_01/results.json',False),
 ('current_recorder_source','coordination/phase_recorder.py',False),
 ('cpu_acceptance_observer','coordination/ACCEPTANCE_HISTOGRAM_CPU_RESULT.json',False),
 ('current_acceptance_observer','coordination/acceptance_histogram.py',False),
 ('math_not_prediction','coordination/DECISION_BOUNDS.json',False),
 ('primary_source_decisions','coordination/PRIMARY_SOURCE_DECISIONS.json',False),
 ('local_cuda_virtual_peer_actual_norm','hopper_moe/virtual_actual_02/results.json',False),
 ('local_cuda_projection_numerics','hopper_moe/projection_plane_01/results.json',False),
 ('local_component_timing','hopper_moe/projection_timing_01/results.json',False),
 ('native_offline_compile','hopper_moe/offline_actual_01/results.json',False),
 ('native_offline_ffi_compile','hopper_moe/offline_ffi_01/results.json',False),
 ('source_push_regression_risks','hopper_moe/PUSH_FUSION_GRID_AND_PDL_AUDIT.json',False),
 ('communication_handoff','hopper_moe/README.txt',False),
 ('communication_final_checkpoint','hopper_moe/FINAL_CHECKPOINT.json',False),
 ('independent_final_handoff','expert_partition/LANE_FINAL_HANDOFF.txt',False),
 ('independent_final_manifest','expert_partition/LANE_FINAL_MANIFEST.json',False),
]
rows=[]
for tier,relative,root_relative in items:
 path=(R if root_relative else P)/relative
 assert path.is_file(),str(path)
 data=path.read_bytes()
 rows.append({'evidence_kind':tier,'path':str(path.relative_to(R)),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
report={'status':'FINAL_CURATED_EVIDENCE_INDEX','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'scope':'Evidence kinds distinguish saved serving results, source/CPU contracts, local CUDA components and native offline compilation. File/check counts are not serving experiment counts.',
 'files':rows,'new_serving_or_H200_execution':False}
(C/'EVIDENCE_INDEX.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'indexed_files':len(rows)}))
