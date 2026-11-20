"""Curated immutable evidence index; no candidate edits or GPU operations."""
from pathlib import Path
import hashlib,json,datetime
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
entries=[
 ('static_hazard_review','wgmma_bra_div_repair_review.json','final four-artifact repaired coverage and static hazard controls'),
 ('static_hazard_review','WGMMA_BRA_DIV_REPAIR_REVIEW.txt','superseding offline scope and CALL limitations'),
 ('static_hazard_review','wgmma_hazard_independent_review.json','initial independent artifact binding review; predecessor static conclusion superseded by repaired receipt'),
 ('ragged_output_n','RAGGED_VERDICT.txt','conditional rejection; numerical pilot only'),
 ('ragged_output_n','ragged_n_pilot.json','12 real-weight shape pilot'),
 ('ragged_output_n','amortization_model.json','charged unknown-reuse sensitivity'),
 ('kv_staging','CONTIGUOUS_KV_CONTRACT.txt','same-byte native-FA3 layout hypothesis'),
 ('kv_staging','contiguous_kv_probe.json','byte and nativeBF16 scaffolds'),
 ('kv_staging','contiguous_helper_gpu.json','bounded lifetime/capacity fixture'),
 ('kv_staging','contiguous_kv_unapplied/incremental.diff','default-off unapplied helper'),
 ('kv_staging','fa3_paged/receipt.json','bounded compile failure retained'),
 ('kv_staging','fa3_contiguous/receipt.json','bounded compile failure retained'),
 ('memory','combined_memory_ledger.json','conditional source lifetime bounds'),
 ('memory','early_projection_unapplied/phase_admission.json','generic explicit admission'),
 ('memory','early_projection_unapplied/order_controls.json','actual scheduler AST ordering'),
 ('memory','early_projection_unapplied/autograd_lifetime.json','actualTorchCPU retained-input proof'),
 ('amendments','early_projection_unapplied/incremental.diff','unapplied early qualification'),
 ('amendments','row_no_grad_unapplied/incremental.diff','unapplied no_grad lifetime fix'),
 ('amendments','row_constructor_cleanup_unapplied/incremental.diff','unapplied handle cleanup'),
 ('amendments','row_constructor_cleanup_unapplied/cpu_constructor_faults.json','exactly-once handle fault controls'),
 ('amendments','row_constructor_cleanup_unapplied/composition_validation.json','private-index composition'),
 ('prologue_veto','prologue_rounding_debug/VERDICT.txt','original4warp arithmetic veto'),
 ('prologue_veto','prologue_rounding_debug/reduction_tree_proof.json','CPU reconstruction of original GPU tree'),
 ('prologue_veto','prologue_rounding_debug/reproducer.json','savedrow observer and original parity'),
 ('prologue_veto','prologue_rounding_debug/decode_extension.json','accepted small-bucket controls'),
 ('prologue_repair','prologue_warp8/REVIEW.txt','separate unreachable-large-shape repair'),
 ('prologue_repair','prologue_warp8/offline_receipt.json','SM90 full compile layout'),
 ('prologue_repair','prologue_warp8/offline_rope_receipt.json','SM90 RoPE compile layout'),
 ('prologue_repair','prologue_rounding_debug/warp8_reproducer.json','savedrow repaired exact'),
 ('prologue_repair','prologue_warp8/byte_matrix.json','42 localbyte cells; no performance'),
 ('bcg','PREFILL_BCG_INTEGRATION_PLAN.txt','precise nonactivated plan'),
 ('bcg','prefill_bcg_source_controls.json','16 actual AST ownership/guard cases'),
 ('fresh_sources','sources/sglang_50801/REVIEW.txt','pinned independent runner audit'),
 ('fresh_sources','sources/sglang_50801/allocation_controls.json','20 actual op allocation cases'),
 ('fresh_sources','sources/nov19_refresh/REVIEW.txt','bounded fresh engine applicability survey'),
 ('terminal_ffn','TERMINAL_FFN_REVIEW.txt','narrow liveness opportunity, unknown latency'),
 ('terminal_ffn','terminal_ffn_liveness_cpu.json','6 actual logits consumer source cases'),
 ('terminal_ffn','sources/nov19_refresh/muse_dflash_config_46c90f.json','actual pinned aux layer IDs'),
]
def record(path):
 b=path.read_bytes();return dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
rows=[];missing=[]
for lane,rel,meaning in entries:
 p=HERE/rel
 if not p.exists():missing.append(rel);continue
 rows.append(dict(lane=lane,path=rel,evidence=meaning,**record(p)))
frozen={
 'scored_parent':('release_2026-11-18/candidate.diff','8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b'),
 'joint_row_n':('research_2026-11-19/prefill/integration_joint/candidate.diff','530e9ccc160398e82386f6e455cdbf80fa4d40f43ec1209b1de7290a5276a8c6'),
 'normal_moe':('research_2026-11-19/moe/normal_package/candidate.diff','80db9ce3d958c58e31be4b64f06442d0ef152493221c3927ad9c43deab68319f')}
frozen_rows=[]
for name,(rel,sha) in frozen.items():
 r=record(ROOT/rel);assert r['sha256']==sha,(name,r['sha256']);frozen_rows.append(dict(name=name,path=rel,unchanged=True,**r))
assert not missing,missing
out=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='Research evidence only; no new2x conclusion or scored submission readiness',
 scope='Local SM86, CPU actual-source controls, bounded offlineSM90 and primary-source research. NativeH200 model/serving proof unresolved.',
 records=rows,frozen_unchanged=frozen_rows,
 nonclaims=['Do not multiply component timings.','Do not combine frozenjoint+normal memory budgets without separate phase fixes.','Do not enable repairedBLOCK32 by increasing16MiB calibration budget.','Do not call upstreamMarlin comparisons gains over scoredHumming.'])
(HERE/'ARTIFACT_INDEX.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(records=len(rows),frozen_verified=len(frozen_rows))))
