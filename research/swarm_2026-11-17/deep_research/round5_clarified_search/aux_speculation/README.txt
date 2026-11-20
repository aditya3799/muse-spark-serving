Round5 clarified search: request-local score export and W4/W8 control

CURRENT: isolated, default off, CPU plus local GPU component qualification. No serving
installer, active SGLang change, remote check or submission.

Concrete artifacts:
- score_prefix_controller.py: actual proposed-path FP32 score margins; seven
  margins plus validity and exact proposed int64 IDs (88 persistent bytes/B1).
  Strong graph/buffer ownership, immutable binding checks, single-use host lease,
  managed CUDA capture/replay tested locally in score_gpu_02.
- score_export_overlay.diff: three added lines in the actual selector sampler;
  absent explicit pre-capture owner attachment, original behavior is unchanged.
  Attachment only supports the B1 graph in this prototype. Production bucket
  registration and mutation hooks are not implemented.
- prefix_decision_bridge.py: actual round4 configuration/resolver context, current
  request rid and epoch/generation checks, restoration on exception, and required
  post-resolver rank agreement. No real distributed collective implementation.
- tests read pinned source AST for lattice, greedy walk, sampler and previous
  prefix resolver. LM projection/top-k, target execution and topology are fixtures.

Findings:
1. Greedy q_out is one-hot. It cannot estimate confidence. Export uses the actual
   predecessor-conditioned lattice row selected by the proposed path.
2. Identical greedy proposals can have arbitrarily different margins. This proves
   score availability, not predictive power. Thresholds here are synthetic and
   explicitly uncalibrated; controller defaults disabled.
3. Agreeing a proposed W4 is insufficient: local topology guard may fall back8.
   A second rank-coherent admission must happen before target. Preserved original
   bridge and result *_PRE_REVIEW_HELD contain this discovered gap.
4. Python capture-phase labels alone cannot prove a write belongs to a CUDA graph.
   Prepared CUDA API owns the graph/context, checks current capture, and binds
   only after success. Actual stream API checks replace caller-key trust for CUDA.
5. Snapshot ownership includes exact proposed IDs to reject mismatched score/path
   packets. Both score and int64 token readbacks incur real D2H/control cost.

Execution/measurement holds:
- No claim that margins predict actual greedy target acceptance. Calibration must
  use authorized own-request scores and measured startup costs, never load or a
  fixed max-running row budget (organizer has not answered that follow-up).
- W means anchor-inclusive verifier rows: W8 contains seven proposals.
- B1-only research admission; no mixed-request width aggregation or grouping.
- Local component CUDA capture passed; real TP8/full target prefix numerics remain held.
- Existing round4 model mutation/recapture hooks are still incomplete.
- Evaluate measured total step cost per committed token; do not scale T(W)
  linearly or project a serving score from this component.

Organizer clarification preserved in round4 ORGANIZER_CLARIFICATION_DRAFTING.json.
Previous whole-serving score remains 1.2237x; no new score is implied.

GPU checkpoint: score_gpu_02 passed82 checks+6 negatives over16 changed replays.
Peakallocated24.4MiB;reserved66MiB (above64 if budget meantreserved).
score_gpu_01 timed out incomplete; preserved. No timing or calibration claim.
See QUALIFICATION_LEDGER.json for exact scope and source hashes.

