"""Reproducible bounds; no timings or target GPU access are implied."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
root=next(a for a in P.parents if (a/'sglang').is_dir())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
status=root/'swarm_2026-11-16/coordination/status_e399c79bb1994630.json'
remote=json.loads(status.read_text(encoding='utf-8-sig'))['result']
world=8;h=8192;e=4096;b=2;beta=(world-1)/world
rows=[]
for m in (512,1024,2048,4096,16384):
 ar=2*beta*m*e*b;ag=beta*m*h*b;rs=beta*m*e*b
 rows.append({'tokens':m,'eligible_under_default_16MiB_pull_cap':m*e*b<=16*1024**2,
 'baseline_AR_plus_unpool_AG_MiB':(ar+ag)/1024**2,'candidate_RS_plus_unpool_AG_MiB':(rs+ag)/1024**2,
 'logical_tail_communication_fraction_saved':(ar-rs)/(ar+ag),'eager_reduced_output_allocation_saving_MiB':m*e*b*(1-1/world)/1024**2,
 'graph_LS_extra_local_output_MiB_vs_stock_inplace':m*e*b/world/1024**2})
score=remote['frontier']['score']
report={'status':'COMPILED_DIAGNOSTIC_NOT_PROMOTED','baseline_score':score,'scored_remote_sha256':sha(status),
 'scored_workload':{'first_turn_tokens':remote['shape']['first_turn'],'new_per_turn':remote['shape']['new_per_turn'],'expected_cache_hit':remote['expected_cache_hit'],'chunked_prefill_tokens_from_startup_log':16384,'acceptance_by_rung':[r['acceptance'] for r in remote['rungs']]},
 'selected_mechanism':'Leave Humming experts/finalization unchanged. Stop the selected custom two-shot pull collective after its original reduction, retain complete row shard, run identical scored unpool GEMM [M/P,E]@[E,H] with FP32 output and BF16 cast, then original row all-gather and decoder boundary.',
 'why_not_generic_reduce_scatter':'Pinned group.reduce_scatter_tensor selects different transport and reduction ordering; its deterministic path explicitly performs original AR+slice. This prototype reuses original LS/MC reduction implementations instead.',
 'kernel_derivation':'Pinned all_reduce_2shot_pull_kernel with unchanged row partition, load_reduce, PDL and both barriers; replace only replicated store with local-output store.',
 'dispatch_contract':['Experimental opt-in only; no active/scored source edit','Exact GroupCoordinator and CustomAllReduceV2 classes','Existing GroupCoordinator FlashInfer precedence and resolved CA method must agree','Existing _pick_config must select TWO_SHOT_PULL; no overrides or enlarged workspace','BF16 contiguous SM90, complete aligned rows, M>=512 divisible TP2/4/8','Inherited model eligibility requires Humming/use_marlin, moe_tp_size==1, no SP/attention DP/CP, no already-reduced output and existing row-projection eligibility','Unsupported paths return None without launching candidate collective or mutating input'],
 'coverage':'TP8/Hopper config commonly selects multicast two-shot in eligible prefills. At illustrative E4096, default16MiB cap allows512<=M<=2048. Full16384-token chunks, ordinary small decode/verify and nondivisible remainders remain baseline. Runtime actual dispatcher telemetry is not available.',
 'bottleneck_estimate':{'geometry_status':'Illustrative geometry matching existing projection fixtures, not newly measured model metadata','world':world,'hidden_width':h,'expert_width':e,'bytes_per_element':b,'formulas':{'baseline_logical_bytes_per_rank':'beta*M*b*(2*E+H)','candidate_logical_bytes_per_rank':'beta*M*b*(E+H)','saved':'beta*M*b*E','beta':'(world-1)/world'},'scope':'Logical bandwidth model, not measured latency or physical NVLS fabric accounting. GEMM, staging, finalization, boundary, synchronization and launch costs are excluded.','rows':rows,'whole_workload_if_eligible_tail_time_fraction':[{'fraction':f,'ideal_speedup_vs_current':1/(1-0.25*f)} for f in (0.05,0.1,0.25,0.5,1.0)]},
 'memory_contract':['No extra weight copy, persistent repack or larger communication workspace','Original full expert output remains until local reduction finishes','Eager stock full reduced output is replaced with M/P rows','LS graph baseline all-reduce is inplace: candidate adds local M/P output instead of saving that full allocation','Existing graph input table registers full original input, retains standard capture/IPC/VMM lifecycle','Both original barriers retained; no rank may recycle source/staging before collective completion','Eager prototype uses cudaMemcpyAsync; stock <=8MiB may use PDL memcpy kernel. Include staging cost in any timing, or match staging before integration'],
 'larger_structural_alternatives':[
 {'name':'Persist row-sharded residual across attention and FFN boundaries','finding':'With unchanged head-parallel attention/router, local attention-boundary output must be gathered before current full router and pooler. Added BF16 hidden-width gather beta*M*b*H exceeds removed expert AR-gather beta*M*b*E when H>E. Keeping boundary output local only once instead requires extra FP32 residual gather. Not selected without changed attention/router data distribution or measured norm savings that exceed communication.'},
 {'name':'Generic LayerNorm-SP switch','finding':'Pinned validator requires TRTLLM/FlashInfer dispatch+combine; Humming has no compatible forward_dispatched path. Do not bypass validation.'},
 {'name':'Shard router token rows with new token-SP','finding':'Prior real gate failed near-tie expert IDs. Preserve full router GEMM geometry; gathering router input is a charged cost.'},
 {'name':'Replace all NCCL expert AR with NVLS RS regardless size','finding':'Not equivalent dispatch/reduction and exceeds original workspace policy; rejected as automatic coverage expansion.'}],
 'three_x_target':{'targets':[{'stock_speedup':t,'required_gain_vs_current':t/score,'minimum_current_time_fraction_eliminated_if_rest_unchanged':1-score/t} for t in (3.0,3.5)],'interpretation':'Per-workload Amdahl illustration, not a prediction for the geometric frontier score. Even treating this eligible tail as all runtime gives only1.333x current under the illustrative bandwidth model. The change is not a3x serving mechanism. Larger gains require dominant expert/attention/scheduling work elsewhere.'},
 'validation_plan':[
 'On authorized multi-Hopper NVLink system, compare exact same stock selected2shot AR output slice vs candidate output bitwise, separately LS/MC and changed-input graphs.',
 'Test cancellation, signs/signed-zero/subnormal/finite extreme BF16 values and arbitrary row order; keep all unsupported dtype/alignment/tail/size fallbacks.',
 'Interleave candidate RS, stock AR, all_gather, changed captures and multiple graph buckets to audit barrier/semaphore reuse and full input-table lifetime; run timeout watchdog.',
 'Compare actual model unpool output and next fused decoder boundary for every rank, including repeat graph replay and source-input ownership; no forced H200 dispatch predicates.',
 'Measure end-to-end eligible tail with same eager staging and balanced repeated controls, plus graph peak memory and allocation across layers; stop if staging erases saving.',
 'Record actual served M histogram, selected AR backend/algorithm and non-overlapped duration before spending scarce remote quota; no existing whole-serving profiler establishes this fraction.',
 'Promote only after independent protocol review and multi-rank numerical/liveness/peak-memory evidence; current single-GPU local hardware cannot establish those gates.'
 ],
 'local_validation':{'cpu_checks':74,'offline_SM90':json.loads((P/'offline_sm90_01/run_receipt.json').read_text()),'GPU_runtime':False,'production_integration':False,'remote_submission':False},
 'source_hashes':{str(f.relative_to(root)):sha(f) for f in [root/'sglang/python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh',root/'sglang/python/sglang/kernels/jit/include/sgl_kernel/vec.cuh',root/'sglang/python/sglang/srt/distributed/parallel_state.py',root/'sglang/python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py',root/'sglang/python/sglang/srt/distributed/device_communicators/configs/custom_all_reduce_v2.py',P/'reduce_scatter_rows.cuh',P/'reduce_unpool.py']}}
(P/'RESEARCH_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'row_cases':len(rows),'report_sha256':sha(P/'RESEARCH_REPORT.json'),'compiler_released':True}))
