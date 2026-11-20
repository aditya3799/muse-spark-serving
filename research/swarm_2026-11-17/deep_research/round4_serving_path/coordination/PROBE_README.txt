COORDINATION PROBES — RESEARCH ONLY

No file here is installed into the active serving checkout. No new serving submission was made. The only confirmed score remains 1.2237x; combined unsubmitted speedup is unknown.

Target argmax capture
--------------------
The existing target greedy path already gathers per-rank maxima instead of full vocabulary logits. This candidate moves that unchanged operation into the target CUDA graph and consumes a persistent token-ID buffer afterward. It is a launch/control-overhead hypothesis, not a new sampling algorithm.

The exact-bucket version is preserved under expert_partition/target_argmax_overlay. target_argmax_overlay_01/probe/results.json executes the actual FullCudaGraphBackend capture/replay/cleanup methods and the overlaid capture wrapper with native softcap argmax, cap20 and local vocabulary25256. It checks changed inputs, all installed bucket confirmations, graph replacement and failed partial recapture. It does NOT instantiate the complete model or runner. TP8 is emulated on one GPU, rank barriers and allocator-scope helpers are stubbed.

The separate padded version is under expert_partition/target_argmax_overlay_v2. target_argmax_padded_01/probe/results.json additionally checks nine real requests inside a ten-request graph:72 valid token rows out of80 captured. NaN/changed padding cannot contaminate the returned leading rows, and inconsistent raw request/token counts reject. This prevents duplicate eager argmax on ordinary padded batches within the narrowly admitted plain fixed-width path. DP, SP/gathered buffers, CP, PP, LoRA, PDMux, ragged layouts and concurrent graph execution remain unsupported.

All ranks must enable, capture, replay and invalidate the candidate consistently. A local fallback does not make divergent collective order safe. Native TP8/NCCL capture, real graph-pool borrowing, full-model correctness and complete-path performance remain untested. Non-greedy graphs still pay for an unused captured argmax. This overlay and the target-prefix overlay are alternatives; they are not qualified for composition.

Optional phase diagnostics
--------------------------
phase_recorder.py wraps actual worker call interfaces after warmup. It samples bounded CUDA event intervals and commit-length histograms without host synchronization inside the recorder. It stores no prompt text, token IDs or request identifiers. It is NOT an inference optimization.

Prefer the existing opt-in DeviceTimer for coarse observation; expert_partition/device_timer_dflash_overlay.diff fixes its missing plain-DFlash draft-runner wiring. The existing timer groups both target and draft under target_verify, so that wiring alone does not separate them. The custom recorder distinguishes draft forward, target forward, argmax/full gather, acceptance, context projection and KV append/materialization.

Use one host thread and one forward stream throughout each sampled step. Call prime() during idle, uncaptured setup. The latest version records a prime-completion event and makes the first sampling stream wait on it; repeated priming is similarly ordered. phase_prime_stream_01/results.json validates cross-stream/chained prime ownership and records cudaStreamWaitEvent with no host synchronization inside the observed step. If prime or installation fails partway, abort/discard the recorder rather than attempting recovery. No recovery protocol is qualified for partially enqueued prime writes.

phase_probe_04/results.json validates the earlier recorder's worker-interface fixture, ordinary changed histograms, nested capture exclusion, slot handling, exception preservation, uninstall and no synchronization inside the observed region. The latest prime-fence delta is separately validated in phase_prime_stream_01; the old results hash must not be represented as the latest source hash.

Nested phase durations are NOT additive. Event intervals include stream idle gaps and may omit unjoined work on other streams. The outer interval includes the diagnostic commit-count transfer. Observer overhead is not qualified as negligible: local controls varied materially. Keep diagnostic runs separate from uninstrumented performance runs. Never infer serving phase fractions from these toy fixtures.

Decision bounds
---------------
DECISION_BOUNDS.json records mathematical limits, not forecasts. Reaching2.5 from1.2237 would require about2.043x additional improvement if all frontier rungs improved uniformly. Under a fixed-work interpretation that is about51.1% runtime reduction; the actual score is a geometric mean, not a single runtime.

An illustrative exact mean commitment of4.1 for width8 does not determine an acceptance histogram. The same mean permits a width4 continuation probability anywhere from2.5% to77.5%. Capping the same boundary at4 gives a mean commitment between2.329 and4.0. Actual published means are rounded, and redrafting after a cap changes future proposals. These bounds explain why an aggregate acceptance number cannot price either candidate.

PRIMARY_SOURCE_DECISIONS.json lists the external references and applicability limits. Published gains from other models, hardware, runtimes or non-speculative baselines are not transferred to this candidate.

CPU-only acceptance observation
-------------------------------
acceptance_histogram.py wraps the existing on_verify_complete_cpu callback, which receives correct-draft counts after their ordinary CPU copy. It imports no tensor library and adds no GPU work or D2H transfer. The actual extracted result-processor test preserves outputs and existing tensor-read counts. It records both marginal correct-draft histograms and joint survivor-count histograms by batch size; the latter captures cross-request correlation needed for compacted continuation cost.

Use only for a fixed width8 DFlash policy. Width8 means one seeded anchor/previous bonus plus seven draft proposals; total step commitment includes the new bonus. Counts are observed before grammar/termination/retraction filtering and are not effective emitted-token counts. This is a lagged diagnostic, not a rank-coherent controller. CPU overhead is unmeasured. Uninstall or otherwise quiesce the callback before serializing snapshot(), because its rows reference is live. No prompts, token IDs or request identifiers are retained.
