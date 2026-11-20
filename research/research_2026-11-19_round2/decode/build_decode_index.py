"""Consolidated evidence index; never a serving score or authorization gate."""
import datetime,hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent
paths=['ROUND2_DECODE_READOUT.txt','freeze_recheck.json','PROLOGUE_STATUS.txt','PROLOGUE_REDUCTION_SCOPE.txt',
 'prologue_freeze_manifest.json','chain_reward.json','component_reward.json','reduction_layout_sm90/receipt.json',
 'SAMPLER_READOUT.txt','sampler_freeze_manifest.json','TARGET_SELECTOR_READOUT.txt','target_selector_freeze_manifest.json',
 'target_selector_cpu.json','target_selector_muse_cpu.json','MUSE_LOGITS_ACTIVATION.txt','muse_logits_activation/receipt.json',
 'DEFERRED_SYNC_READOUT.txt','deferred_sync_freeze_manifest.json','deferred_sync_host_cost.json',
 'DFLASH_SYNC_AUDIT.txt','ATTENTION_BACKEND_AUDIT.txt','RING_PREFETCH_AUDIT.txt','RING_DEFAULT_OFF_SUPPLEMENT.txt',
 'ring_default_off/manifest.json','PR51393_READOUT.txt','pr51393/audit.json','pr51393/manifest.json',
 'FLASHINFER_NOV_READOUT.txt','flashinfer_nov/manifest.json','cluster_softcap/PLAN.txt','cluster_softcap/cpu_protocol.json',
 'cluster_softcap/frontend_failure_v1.json','cluster_softcap/v2/frontend_failure_v2.json',
 'cluster_softcap/v3/stable_preflight.json','cluster_softcap/v3/api_correction.json','cluster_softcap/v3/gluon_prototype.py',
 'VENDOR_CFG_REVIEW.txt','MOE_BOUNDARY_REVIEW.txt','final_static_cross_review.json','donor_cfg_independent_review.json']
out=dict(snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 baseline_sha256='8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b',
 baseline_score=1.2363,conclusive_2x_serving_evidence=False,scored_submission_requested=False,
 remote_jobs_launched_by_this_lane=False,all_integrations_default_off=True,all_integrations_unapplied=True,
 statuses=dict(prologue='Local component positive; wider chain1.00208x neutral for material gain; scope correction retained.',
 draft_sampler='Exact local/Gloo and actual-class proof; no NCCL/H200/serving gain.',
 target_selector='Default-off frozen; captured local component only; eager controls fail; Muse dispatch supplemental proof.',
 deferred_sync='Frozen base-class prototype INELIGIBLE for Muse; host-cost negative caveat.',
 cluster_softcap='CPU protocol/source plan only. Two frontend failures retained; final API-correctedv3 compiler neverstarted due memory stability gate.',
 vendor_wgmma_audit='Six final artifact-bound static screens pass repaired vendor successor coverage and reachability; original failures preserved. Not hardware numerical/memory/performance proof.',
 donor51393='New source pinned; T1/2auto,max4 excludes fixedDFlashT>=8; BF16 boundary differences.',
 model_precision='Pinned Muse forcesFP32 LMhead; actual checkpoint vocab/softcap still unauthenticated.'),
 artifacts={name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in paths},
 required_before_serving_promotion=['ActualH200 numerical execution','RealTP8 collective ordering and latency','Full model startup/capture memory','Native dispatch activation','Full official frontier independent holdout'],
 reminder='No local component ratios are summed or multiplied into a serving score.')
(here/'DECODE_INDEX.json').write_text(json.dumps(out,indent=2));print(out['snapshot_utc'],len(paths),'artifacts indexed')
