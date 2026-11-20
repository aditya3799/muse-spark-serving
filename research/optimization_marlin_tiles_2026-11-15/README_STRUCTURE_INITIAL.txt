MARLIN EXPERT-PARALLEL TILE EXPERIMENT 01
Created 2026-11-15. New phase, separate from the closed mapping notebooks.

Purpose
Compare the submitted candidate's existing local-density tile heuristic with a
global-density heuristic and fixed blocks 8/16/32/48/64. No candidate selected yet.
The baseline source must match SHA256:
18438f083871ec322bdc409690f8b2674cef5381e1e786176eae685dddea2c0f
The source is read, checked and compiled into isolated functions in memory.
Only the selector changes. The serving checkout and submitted patch are untouched.

Files
tile_study.py: synthetic CPU route partitioning, padding and conservation checks.
  Optional CUDA graph tests use PyTorch counts, NOT SGLang alignment or GEMMs.
01_marlin_tile_structure.ipynb: self-contained copy for Kaggle/Colab; no installs.
marlin_bench.py: Linux SM80-SM90 BF16/NVFP4 complete local expert-path benchmark.
  Real alignment, both Marlin GEMMs, activation and Muse finalization are executed.
  Synthetic packed weights are prepared by the pinned SGLang preparation code.
test_experiment.py: geometry counterexamples, corrupt-table detection and AST isolation.
analyze_structure.py: maximum-rank structural comparison for the local default study.
local_structure/: measured local structural results and checksums.
local_preflight/: actual Windows preflight result; benchmark unsupported on Windows.
LOCAL_REVIEW.txt: what passed, hardware limitations and next decision.

Run structural study (CPU is enough)
  python tile_study.py --out results
  python tile_study.py --out results --cuda-route-checks
  python -m unittest -v test_experiment
Set TILE_SGLANG_ROOT for source-isolation tests outside the original Windows path.

Notebook
Upload 01_marlin_tile_structure.ipynb and Run All. CPU is sufficient; use T4 only
if you want the optional graph-count check. Download marlin_tiles_01_*.zip.
Place it in this experiment's notebook01_results folder and stop the GPU session.
These checks already passed locally; repeating on T4 is optional and cannot
establish which Marlin tile is faster. Do not use T4 GPU time for Marlin timing.

Real-kernel smoke run (existing Linux SGLang environment, no automatic installs)
  python marlin_bench.py --sglang-root /path/to/serving/sglang --out preflight
  python marlin_bench.py --sglang-root /path/to/serving/sglang --out smoke \
    --mode bench --tokens 8 9 33 128 --samples 10 --repetitions 20
Default synthetic H=256/I=256 is a correctness/integration smoke test, NOT Muse
serving geometry. Successful import preflight does not prove JIT execution works.
The full GPU benchmark has NOT been executed locally. Dependency/JIT/numerical
failures are saved with tracebacks; a failed gate produces no recommendation.

Performance run, after smoke passes
Use confirmed model expert dimensions via --hidden and --intermediate (multiples
of 128 in this harness), keeping --global-experts 256 --local-experts 32 --topk 8.
Include --tokens 1 7 8 9 31 32 33 127 128 129 255 256 257 1024 4096 as memory allows.
Start with bounded subsets, then expand only to resolve uncovered shapes.
Run rank 0 and a representative other shard using --rank. Independent rank runs
do not substitute for simultaneous eight-rank latency/communication measurements.
All default routing scenarios use unique within-token choices, with repeated
choices across tokens. Add adversarial duplicate slots only as correctness tests
if a future dispatcher can emit them; do not treat them as representative timing.

Measurement
Both policies use identical weights and input data. Includes baseline-vs-baseline
noise control. Alternates AB/BA CUDA-event samples, records raw values, and keeps
eager and graph results separate. Changes routes, activations and scores in place
before comparing graph replay to eager. All-remote and inactive cases must return
zero. Each candidate is checked against the submitted baseline, default rtol/atol
0.01/0.01; exact equality and maximum absolute differences are also recorded.
Those tolerances are an experimental screen, not permission to change precision
and not full-model greedy-token validation. No approximations are introduced.

Memory
Records weight-preparation and graph-capture peaks. Pair peaks include two captured
graphs and are NOT the candidate's deployment overhead. Model-wide loading,
capture and serving memory still require independent target-device measurement.

Decision gates
1. Structure conservation and isolated-method checks pass (done locally).
2. Real BF16/NVFP4 complete-output and changed-input graph checks pass (pending).
3. Repeated full-path H200 gain holds for relevant shapes and skew, above noise.
4. Integrate only a justified generic shape/device policy; no load-label dispatch.
5. Review full-model correctness, TTFT, throughput and memory in an authorized
   serving check. No new submission is created by any experiment script here.
Sorted-EP support is a separate future experiment; it is not changed here.

Limitations
No checkpoint weights, distributed collectives, scheduler, router or full serving
execution are included. Balanced density is a hypothesis: hot ranks can have much
more local work. Smaller blocks can reduce padding while increasing weight loads.
No T4/RTX timing, padding ratio or single-layer ratio is a serving-speed prediction.
