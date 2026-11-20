ROUND 4 ATTENTION COLLECTIVE RESEARCH — 2026-11-17

This lane contains isolated experiments. Nothing is imported by active serving code.
No submission, remote check, paid compute, or TTS process change was performed.
The 2.5x serving target is not established by this work.

CANDIDATES AND THE ACTUAL BASELINE

The frozen scored Muse attention o_proj is a BF16 row-parallel projection followed
by the attention-TP all-reduce and the post-attention norm/residual/pre-FFN norm.
Its audited methods match the active checkout AST; other model paths differ.
The current scored startup log reports V2 multicast=True, pull=True and 18.01 MB
symmetric memory. Per-call algorithm selection remains source-conditional:
  H8192, TP8: M8 is exactly 128 KiB, the inclusive one-shot push threshold.
  M32/128/256 fit the multicast two-shot region. M753 fits the 16 MiB pull cap.
  M3012 does not fit that cap and must not be admitted as this candidate.
The candidate does not apply to every layer, collective, prefill size, or backend.

1. Small M one-shot push + the unchanged norm body.
   attention_ar_norm.cuh uses the pinned Lamport sanitization and rank-ordered
   reduce_vec, BF16 reduction output boundary, then the original norm arithmetic.
   Actual attention specialization has post_weight=False, pre_weight=True,
   gated residual, post eps 1e-8/pre eps 1e-5 from config defaults, router=True
   only in the small-M branch. Actual runtime eps and cached coefficients must
   be read from the model; defaults are not a fresh checkpoint inspection.

2. Broad M multicast workspace handoff.
   mc_workspace_handoff.cuh retains the native two-shot multicast kernel and
   input staging, returns its ordinary-VA local workspace to the original norm,
   and eliminates the output copy/allocation. mc_ready_plane_handoff.cuh also
   accepts an already completed projection in that local plane, eliminating
   input staging. It preserves stock multimem BF16 load-reduce with FP32
   accumulation; it does not replace it with a software rank sum.
   A single MC+norm kernel was rejected: per-block peer barriers do not join all
   local producing CTAs, while one norm row can span two MC-producing blocks.

EVIDENCE TIERS

Local SM86 numerical evidence:
  virtual_ar_02: 45 finite bitwise checks for the GENERAL norm fixture with
    post_weight=True. This is not the target attention specialization.
  virtual_actual_02: 45 finite bitwise checks with actual post_weight=False
    specialization and gate formula. M1 tests all eight virtual ranks via a
    cooperative launch; M2/M8 prepopulate peer planes and test consumers only.
    These are not eight-GPU visibility, progress, or performance tests.
  projection_plane_01: 20 finite bitwise F.linear versus mm.out comparisons,
    M32/128/256/753, K1024/N8192; eager and changed-input/weight graph generations.
    This qualifies the local producer operation only, not H200 cuBLAS choices.

Offline SM90a evidence, no CUDA context:
  offline_actual_01/push_actual: original actual norm 188 registers versus
    fused push/norm 233 registers, 256 threads, 64 B static shared, no spills.
  offline_actual_01/mc_actual: actual broad norm 175 registers/256 threads for
    small M and 56 registers/512 threads for large M; MC 18 registers, no spills.
  The handoff kernel retains multicast arithmetic/SYS ordering in generated
    SASS. Compilation does not prove multi-GPU visibility or throughput.

Local producer timing, shared GPU with TTS:
  projection_timing_01 includes warmup, A/B/B/A, opening/closing A/A, B/B and
    first-half/last-half drift checks. M32, M128 and M256 fail stability gates;
    their apparent gains are unqualified. M753 passes the recorded controls:
    F.linear+copy 1141.71 us, mm.out 920.88 us, difference 220.83 us.
  This is an absolute component result on SM86. It is not collective timing,
    H200 timing, serving speedup, or a benefit to multiply with previous work.

CPU/source evidence:
  Arithmetic negatives demonstrate that moving norm or projection across AR,
    dropping its BF16 boundary, or changing rank reduction trees changes bits.
  Protocol models include rejected early overwrite/partial poll/missing epoch
    bumper controls. Models are not CUDA memory-model proofs.
  attention_handoff_guard.py has CPU mocked admission/revalidation tests. Those
    check Python control flow, not real runtime import, CUDA, or communicator
    behavior. The independent review also identifies capture/replay limits.

CALLER ADMISSION AND OWNERSHIP

attention_handoff_guard.py is default-off. It requires H200 SM90, current device,
TP8 and attention TP8, no DP/SP, the exact V2 communicator owned by the projection
group, no higher-priority group collective path, no algorithm override, and the
CURRENT _pick_config result TWO_SHOT_PULL + use_multicast + use_graph=False.
It validates actual plane capacity/current pointer, Torch/AUTO unquantized
projection, no alternate reduction/output flags, cached actual norm coefficients,
and all live tensor shapes/dtypes/devices/nonaliasing before writes. Prepared
plans revalidate configuration and tensor pointer/layout/eps before immediate use.

The helper lock excludes only users of this helper. It does not prevent foreign
stock communicator calls. All-rank admission and exclusive serial attention
callsite ownership are explicit caller obligations, not inferred guarantees.
The frozen attention callsite is serial, but whole-worker multistream ownership
is still unvalidated. The shared plane must remain exclusively borrowed until
norm consumption finishes; no returned raw AR view may escape to later work.
The next same-stream collective's entry barriers protect a slower peer's local
norm only if every communicator user obeys this sequence and plane lifetime.

Python preparation and locks run during capture, not later graph replays.
Captured graphs require immutable plane/module/config/parameter layout and
external graph invalidation before mutation. Replay-time foreign-use exclusion
is not supplied by this helper. The future native gate is eager-only for now.

PDL LIMIT: candidate entries use PDL=false, while stock Hopper may use PDL=true.
Removing a copy can lose overlap and regress. A future H200 benchmark must
compare native baseline, matched PDL-off baseline and candidate; copy count alone
is not a speed claim. Input/output alias and next-plane reuse checks stay required.

REPRODUCTION

Run from this directory. CPU-only commands, no CUDA import:
  python audit_collectives.py
  python audit_frozen_attention.py
  python model_push_protocol.py
  python model_push_fine.py
  python audit_ar_boundaries.py
  python test_attention_handoff_guard_cpu.py
  python h200_mc_gate.py
The last command only validates the source ledger and prints the held native gate.

Existing local numerical receipts can be reproduced only under a new serialized
GPU/compiler lease and the 512 MiB free-memory reserve, using Ubuntu Python:
  /home/aditya/.venvs/okc-marlin-tiles/bin/python probe_virtual_attention_actual.py --out NEW_RUN_DIR
  /home/aditya/.venvs/okc-marlin-tiles/bin/python probe_projection_into_plane.py --out NEW_RUN_DIR
Preserve existing output directories. Exact offline compiler argument arrays are
in offline_actual_01/{push_actual,mc_actual}/command.json; use a new output path.

Future native numerical gate (NOT RUN; no H200 available):
  torchrun --standalone --nproc-per-node=8 h200_mc_gate.py --run-h200 --out NEW_H200_DIR
It refuses non-H200/non-SM90/non-local-eight-rank execution, checks actual MC
selection, uses a CPU agreement group before dependent collective branches, and
compares stock, PDL-off, staged-handoff and direct-producer output/norm boundaries.
The future gate explicitly compares native stock norm dispatch (PDL=true on
H200) against exported PDL-off norm outputs, as well as stock/PDL-off raw AR.
It prepends the pinned checkout Python path and verifies imported module paths.
These final Python gate additions are syntax/source reviewed only; the compiled
FFI exports/header are unchanged from offline_ffi_01.
It does not test full serving, graph replay, other communicator users or timing.
All six native gate FFI exports, including local-address helper, compiled to a
full SM90a object in offline_ffi_01 (77.46s). Linking, loading, FFI invocation and
actual H200 execution remain pending. Unused-variable warnings are retained.

Retained failures: initial SM86 build omitted SGL_CUDA_ARCH; first actual-branch
launch had a UTF8/BOM syntax error before CUDA. Later successful receipts are
separate. Do not reinterpret failed or general-fixture receipts as target proof.

PUSH GRID DOWNSIDE
PUSH_FUSION_GRID_AND_PDL_AUDIT.json records the exact warp-major ownership.
M8 stock push uses132 nonempty CTAs of128 threads; fused norm/push uses8 CTAs
of256 threads. M1 changes32 nonempty stock CTAs to1. The fused233-register
kernel allocates approximately61,440 registers/CTA after warp rounding; the
actual original norm188 registers rounds to49,152 registers/CTA. Both are at
most one256-thread CTA/SM under the64K register model. This is a possible
regression mechanism, not a favorable occupancy or latency estimate.

PRIMARY REFERENCE AND STREAM-SCOPE NOTES
The native multimem operation and ordinary/local VA distinction were checked
against the CUDA13.0.2 PTX and CUDA Driver multicast documentation:
https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red
https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MULTICAST.html
WORKER_STREAM_OWNERSHIP_SOURCE_SCOPE.json pins the scheduler forward stream and
separate graph/PDMux paths. It is intentionally a partial source audit. The
helper rejects TBO/PDMux/CP in addition to DP/SP; whole-worker ownership is held.

FUTURE GATE LIMIT
The held H200 fixture uses CPU agreement and synchronized numerical checks
between stages. This is deliberately a numerical gate; it does not stress
uneven rank progress or asynchronous next-generation plane reuse. Those need
a separate native lifetime/capture test before any serving integration.
