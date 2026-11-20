import hashlib,json
from pathlib import Path
P=Path('swarm_2026-11-17/deep_research/round4_serving_path');C=P/'coordination'
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
timing=read(P/'hopper_moe/projection_timing_01/results.json')
qualified=[r for r in timing['rows'] if r['qualified']]
assert len(qualified)==1 and qualified[0]['M']==753
metadata=read(P/'aux_speculation/metadata_gpu_02/probe/results.json')
aux=read(P/'aux_speculation/fullshape_aux_gpu_02/probe/results.json')
padded=read(C/'target_argmax_padded_01/probe/results.json')
assert metadata['status'].startswith('PASS') and aux['status'].startswith('PASS') and padded['status'].startswith('PASS')
assert len(aux['checks'])==64 and all(r['bitwise'] and r['finite'] for r in aux['checks'])
assert aux['peak_allocated_bytes']<=900*2**20
q=qualified[0]
findings={'status':'RESEARCH_COMPLETE_SERVING_GAIN_UNVERIFIED','confirmed_score':1.2237,'confirmed_job':'e399c79bb1994630','new_serving_speedup':None,'combined_projected_speedup':None,'target_2_5x_established':False,
 'local_timing':{'scope':'SM86 producer only, shared GPU; no communication/H200/serving inference','M':753,'K':1024,'N':8192,'baseline_us':q['baseline_linear_copy_us'],'candidate_us':q['direct_mm_out_us'],'saved_us':q['saved_us'],'stage_time_reduction_fraction':q['saved_us']/q['baseline_linear_copy_us'],'rejected_shapes':[r['M'] for r in timing['rows'] if not r['qualified']]},
 'candidates':[
  {'name':'Target-only verification widths','evidence':'Actual CUDA metadata kernels and width-specific graph buffers; CPU source-level worker integration','readiness':'Default-off scaffold; actual target model/FP8 attention/TP8/graphpool/lifecycle/cost held'},
  {'name':'Segmented continuation of same width8 candidate block','evidence':'CPU chain/slot/attention-mask/rank/accept-buffer contracts plus fullsize same8B FC/native RMS staging component','readiness':'Algorithm/source proposal; no full segmented target execution','known_cost':'Retains original full8 auxiliary projection/append A8 plus staging; repeated target passes may reread weights','known_hazard':'Shared accept_len scratch requires private stage decisions and single final publication'},
  {'name':'Attention workspace handoff','evidence':'Corrected local norm/virtualpeer boundary; local producer equality; native SM90a compilation and caller guards','readiness':'H200 actual execution, wholeworker plane ownership and performance held','regression_risks':['PDL-off vs stock PDL','smallM push CTA distribution reduced','helper lock not global communicator exclusion']},
  {'name':'Target argmax graph tail','evidence':'Exact existing arithmetic, actual backend lifecycle, native local cap20/vocab25256, paddedB9to10 poison gate','readiness':'RealTP8/NCCL, graphpool borrowing, fullmodel and fullpath timing held','composition':'Prefix overlay rejects this hook; no combined qualification'}],
 'diagnostics':['Existing opt-in DeviceTimer DFlash wiring fix; target/draft categories still aggregate','Bounded optional subphase recorder; observer overhead unqualified','CPU-only existing acceptance callback observer; no additional GPU/D2H work, host overhead unmeasured'],
 'negative_findings':['Reordering alone gives no throughput benefit in additive scheduler model','Rank-local wallclock control can diverge collective order','Saved means/percentiles do not identify phase fractions or survivor histogram','Published external speedups not transferable to this already-speculative baseline'],
 'preservation':{'active_serving_edits':False,'submission_or_remote_check':False,'paid_compute':False,'tts_actions_by_swarm':False,'tts_final_observation':'Previously tracked container exited137 at18:27:12UTC; OOMKilledfalse. Intent clarification pending; no causal attribution or uninterrupted-uptime claim.'}}
(C/'RESEARCH_FINDINGS.json').write_text(json.dumps(findings,indent=2)+'\n')
print(json.dumps({'status':findings['status'],'qualified_stage_saved_us':q['saved_us'],'fullshape_checks':len(aux['checks']),'metadata_replays':metadata['replays'],'padded_checks':len(padded['checks'])}))
