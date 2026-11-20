"""Current evidence index; no kernel/remote execution or promotion."""
from pathlib import Path
import hashlib,json,datetime
here=Path(__file__).resolve().parent;serving=here.parents[1]
def receipt(path,scope,status):
 p=here/path
 return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope=scope,status=status)
items=[
 receipt('bridge_cpu.json','Independent WGS-to-Humming/normal byte and scale pack oracles','PASS correctness only'),
 receipt('bridge_gpu.json','24original-pack GPU/changed-weight/graph converter cases','PASS SM86 correctness'),
 receipt('facade_contracts.json','Torch-free facade import and14metadata/lifetime negatives','PASS CPU contracts'),
 receipt('facade_gpu_930239.json','15E2 fullwidth FFN cases','Correctness only; oversubscribed, memory/timing INVALID'),
 receipt('facade_memory.json','E32M4097 phase-separated complete FFN','PASS byte hashes; single-layer delta864MiB, not serving peak'),
 receipt('fallback_930293.json','Restore canonical weights then complete FFN versus scored Humming','REJECT as decode optimization; conversion cost dominates'),
 receipt('bridge_tiled_cpu.json','Independent tile-aligned address map','PASS50,331,648positions'),
 receipt('bridge_tiled_gpu.json','24independent tiled conversion/replay cases','PASS SM86 correctness'),
 receipt('tiled_holdout_evaluation.json','Improved versus original restore+sameFFN, one predeclared holdout','3/8local controls pass;5remain unqualified; NOT canonical-parent win'),
 receipt('hybrid_traffic_memory.json','Repack traffic/owned workspace algebra','Conditional geometry only, no H200 timing'),
 receipt('register_dequant_cpu.json','Exhaustive finite coefficient and mixed packed inputs','PASS CPU arithmetic oracle'),
 receipt('register_dequant_gpu.json','All code/scale pairs and mixed packed inputs','PASS SM86 arithmetic; no Hopper execution'),
 receipt('finite_scale_scan_gpu.json','Raw and WGS-resident finite scale classifier','PASS99cases'),
 receipt('finite_frozen/manifest.json','Finite-only WGS source/SM90 build/SM86 coefficient proof','FROZEN; disabled; no H200 runtime'),
 receipt('finite_lifetime_v2/manifest.json','Canonical lease/TOCTOU/weakref/reentrant retirement contract','FROZEN;21CPU contracts and independent review; owner integration pending'),
 receipt('split_readiness_frozen/manifest.json','Split barriers and consumed hint variants','FROZEN; source/protocol/SM90 compilation only'),
 receipt('split_wait_dataflow.json','All32entry and32retry SASS sites','Dependency places full decoded group before retry branch; no speed claim'),
 receipt('h200_plan_frozen/manifest.json','Disabled plan/schema/receipt consistency evaluator','FROZEN;76negative synthetic tests; NEVER authorizes submission'),
 receipt('CURRENT_SOURCE_LEADS.json','Six pinned primary-source applicability studies','Existing tricks/exclusions separated from new hypothesis'),
 receipt('sources/sglang_cookbook_manifest.json','Latest SGLang Hopper recipe follow-up','MXFP4/FP8 path distinction; not new NVFP4g16 backend'),
 receipt('target_timing_review.json','Independent review of root target-select measurement controls','Synthetic counterexample; root subsequently repaired gate'),
]
frozen={
 'release_2026-11-18/candidate.diff':'8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b',
 'research_2026-11-19/moe/normal_package/candidate.diff':'80db9ce3d958c58e31be4b64f06442d0ef152493221c3927ad9c43deab68319f',
 'research_2026-11-19/moe/normal_package_launcher_fix/candidate.diff':'8b78849532277aef92994e1e00697c04d00fcd13fed7eafbe19350e8a5b898ab'}
for path,expected in frozen.items():assert hashlib.sha256((serving/path).read_bytes()).hexdigest()==expected,path
r=dict(updated_utc=datetime.datetime.utcnow().isoformat()+'Z',research_window='10:32:37..13:32:37UTC',research_window_complete=False,
 strongest_new_lead='Finite-only exact WGS register coefficients;96register/no-spill SM90 build. Split-ready/hint scheduling is a separate unexecuted extension.',
 conclusive_2x_serving=False,scored_submission_recommended=False,no_remote_jobs_launched_by_this_lane=True,
 strongest_negative='WGS-to-native restoration is compatibility, not a decode speedup. Old WGS baseline already regressed serving; beating it alone is insufficient.',
 evidence=items,frozen_prior_artifacts_verified=frozen,
 unresolved=['Actual H200 arithmetic/progress/latency','Real route holdouts across model layers/ranks','Full model joint peak<=2GBextra','New grouped-prefill submitted job result owned byroot','Canonical owner/reload/graph integration','Full serving token/TTFT/TPOT/frontier quality and independent2x'],
 preserved_failures=['Initial missingCUDAenvironment','Initialoversubscribed facade timing/memory','Cached metadata mismatch corrected beforephase-separated E32test','BOM compiler parse failure','Initialsplit compileheadroom preflight refusal','First split SASS partially defeated source ordering','Tiled pilot/noisy5of8holdout controls'],
 lifetime_limits=['.data/rawCUDAwrites require explicitretirement; no automaticdetection','Facade postscale tensor replacement requires new resident/facade aftergraphretirement','Finite proof declines inference tensors withoutversion metadata','Unknownstream/TBO/SBO concurrentintegration notqualified'])
(here/'ROUND2_MOE_EVIDENCE.json').write_text(json.dumps(r,indent=2));print(json.dumps(dict(items=len(items),prior_frozen_hashes_verified=len(frozen),conclusive_2x=False)))
