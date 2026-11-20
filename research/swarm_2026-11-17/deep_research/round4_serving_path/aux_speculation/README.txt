ROUND 4: TARGET VERIFICATION / REQUEST LIFECYCLE RESEARCH

Outcome
Two structural alternatives have useful partial correctness evidence, but neither is enabled or qualified for serving. The confirmed whole-serving score remains 1.2237x. No new score, 2.5x/3.8x projection, H200 result, submission, remote check or paid compute was produced here. Active SGLang files and external TTS were untouched.

1. Fixed draft8, capped target verification width
The draft still generates the original eight-token block. Only target verification uses W; accepted output, bonus, target KV writes, draft KV materialization and result stride change together. This changes iteration boundaries and subsequent proposals, so saved baseline acceptance means cannot predict its actual output-per-step distribution.

- eager_prefix_overlay.diff: inert eager-only precursor, preserved.
- graph_prefix_overlay.diff: separate default-off factory/scope scaffold. It builds a dedicated FA3 backend and real DecodeCudaGraphRunner(width=W) at DFlash.init_cuda_graphs entry, after target startup capture and before draft TP scope / post-capture KV resizing. It scopes preparation through target, acceptance, publication and auxiliary consumption, then restores the default state.
- GRAPH_PREFIX_OVERLAY.json and TARGET_GRAPH_FACTORY_CPU.json: exact guards, source hashes and holds. CPU factory passed five switches and27 negative controls. Unsupported DP/CP/PP/TBO/PDMux/LoRA/Mamba/other capture tails are rejected.
- GRAPH_SCOPED_WORKER_TAIL_RESULT.json: 180 source-extracted cases,36 default8 reference comparisons,12 fallback cases,12 actual source trimming cases,12 padded-output rejections and12 absent-host-length cases. Models and graph runner are explicit fakes; actual worker control and FA3 metadata methods are extracted.
- metadata_gpu_02/probe/results.json: real pinned normal_decode_set_metadata and translated FULL/SWA read-table CUDA kernels;12 graphs across W2/4/6/8 and BS1/3/4,48 changed replays,240 exact comparisons. Peak152064B allocated,4MiB reserved. This uses GraphFacade, NOT actual FA3 attention or full DecodeCudaGraphRunner. Later stream-binding/invalidation manager changes are CPU-only.

Startup/lifecycle limits
New state allocation is not memory-neutral. Meta-device source accounting gives about345MiB extra FA3 metadata plus36MiB input/aux tensors for W2/4/6 at BS32/context131072, excluding graph workspace, cubins and allocator overhead. These allocations reduce post-capture KV headroom even though weights and KV pool objects are shared. Actual startup peak and the +2GB rule remain untested.

ModelRunner.forward_stream exists before capture. Scheduler.schedule_stream is created later for non-overlap execution; the manager binds to the first serving stream and rejects rebinding. The frozen manager comment saying scheduler creates its serving stream later refers to schedule_stream, not ModelRunner.forward_stream.

Explicit invalidation APIs exist, but generic production weight-update/recapture hooks are NOT installed. Object identities do not detect in-place parameter mutation or tensor.data replacement. Disk and distributed weight loaders do not all pass through begin_weight_update. Rebuild/retirement must be integrated before promotion. Default-off scaffold must stay held.

2. Segmented verification of the SAME draft8 proposal
First verify inputs c0..c3. Continuation requires FOUR comparisons p0..p3 against c1..c4; ordinary width4 acceptance checks only three and is insufficient. Survivors verify c4..c7 with prefix length L+4 and original tail slots. No redraft or intermediate publication is allowed. Under model equality, final acceptance/bonus/next draft state match baseline.

Preferred auxiliary strategy: canonical full[B,8,40960] BF16 staging,20MiB at B32. Copy first auxiliary rows before the next target overwrites its shared output, scatter survivor tail rows to original request positions, zero unused tail rows, then call the ORIGINAL full8 append once. This retains A8 projection/materialization cost and adds memset/copy/merge cost; it avoids changing the replicated FC's matrix shape.

- SEGMENTED_VERIFY_CPU.json:1536 ideal prediction-oracle cases,13824 assertions, all survivor counts0..3, stable/reversed compaction, exact accepted/bonus/output/KV slot bookkeeping. Not a model numerical proof.
- SEGMENTED_AUX_STAGING_CPU.json:64 reduced-geometry CPU cases,1024 exact finite comparisons through extracted FC, native CPU RMS, five KV projections and append control. RoPE and cache writer are explicit stand-ins.
- fullshape_aux_gpu_02/probe/results.json: actual extracted base DFlashDraftModel.project_target_hidden (inherited by DFlash2), full40960→8192 replicated BF16 FC, and actual RMSNorm.forward_cuda standard branch invoking installed sgl_kernel native RMS. B1/3/8/32, eight changed generations each,64 strict finite/byte-equal rawFC and normalized retained-row comparisons, all commit lengths1..8 and32 asserted FC/norm parameter mutations. Peak827606016B (~789MiB), under900MiB budget with512MiB reserve. No target model or full five-layer GPU KV append. Installed sgl_kernel0.4.10 and loaded binary/wrapper hashes are recorded; no judge-binary equivalence claim.
- SEGMENTED_ATTENTION_MASK_CPU.json:160 causal/SWA cases,1280 interval equalities using pinned bottom-right masking formula. Standard FA3 cached MHA already stores current KV before reading it, so segmentation adds no new raw-to-FP8 cache boundary by itself. Projection values and automatic SplitKV arithmetic can still differ.
- SEGMENTED_RANK_SCHEDULE_CPU.json: actual SpecTpSync parser/enabled/sync with fake eight-rank bus,32 cases/680 assertions. Startup configuration consensus is required. Existing greedy prediction broadcast can make survivor order identical when enabled; host survivor-count availability and a second collective sequence remain explicit costs. No actual NCCL or graph-conditional branch was tested.

Cost / next decisive gate
Use measured T(W,B,L,routes), not T8*W/8. Smaller M may read the same experts and use the same tile; segmented mode may read them twice. With full8 auxiliary staging, compare T8 against T4(B)+I(S>0)*T4(S)+staging+decision+merge. D8 and A8 remain fixed floors. Saved mean acceptance4.0–4.1 does not reveal survivor distribution or request correlations. Full-model greedy/targetKV/draftKV equality and clean end-to-end cost must pass before promotion. No new hardware work is authorized by this handoff.

Preserved failures
metadata_gpu_01 failed a table oracle after the fixture randomized reserved v2p[0]; the source requires a zero sink. The corrected receipt is separate. SEGMENTED_AUX_FAILURE_01.json preserves a B1 .contiguous() alias in the CPU oracle; cloning the independent fake shared output fixed the fixture. fullshape_aux_gpu_01 failed before numerical checks because installed package dist-info was absent; retry used module version plus native binary provenance. No numerical tolerance was relaxed.

Primary context
https://github.com/sgl-project/sglang/issues/44678
https://arxiv.org/html/2605.00342
Prefix truncation, precaptured target widths and profiled cost selection have prior art. No novelty claim or published percentage is transferred to Muse. SHORTLIST_AUDIT_ADDENDUM.json closes the misleading metadata-glue startup-log lead. ARGMAX_ADAPTER_INDEPENDENT_REVIEW.json records the separate source review for the parent/Expert lane.

QUALIFICATION_LEDGER.json is the qualification index. Raw receipts, snapshots, failures and tests are retained. No more GPU/compiler experiments are queued in this lane.
Late independent-review ownership finding
SEGMENTED_ACCEPT_BUFFER_CPU.json tests the actual pinned _ensure/_next_accept_bonus_buffers methods. _accept_len_buf has one shared allocation; compact tail acceptance can overwrite first-stage decisions for a different original request. Use dedicated stage1 decisions and private tail workspace, then reserve the final full-batch output/commit/bonus ring once and publish once. Simply calling stock _accept_block twice is held. The production segmented result must also merge per-request metadata; compact-tail target_out cannot stand in for the full batch. Requested routed-expert/indexer diagnostics need exact merge or fallback.

Future memory option (source-only)
FUTURE_MEMORY_OPTION.json identifies unused decode/draft-extend buffers in dedicated target-only backends. About82MiB/state could be omitted in a future guarded initializer. This is NOT implemented or runtime-qualified; the current approximately381MiB extra logical estimate remains unchanged. Width-constant cu_seqlens_q must never be shared across different W. The approximately135MiB future subtotal is arithmetic only, excluding graphworkspace and allocator costs.
