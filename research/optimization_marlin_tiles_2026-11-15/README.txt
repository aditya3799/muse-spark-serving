LATEST: scored job468701af35b84e07 passed at1.0111x. See SUBMISSION_STATUS.txt. New research is in ../research_iteration_2026-11-15. Historical profiling snapshot follows.

MARLIN EXPERIMENT BUNDLE

Start with SESSION_REVIEW.txt. SESSION_RESULTS.json records the full evidence index.
The selected unapplied cumulative patch is candidate_compact32_01/candidate_unsubmitted.diff.

Runtime: WSL Ubuntu, /home/aditya/.venvs/okc-marlin-tiles/bin/python.
Launcher: run_local_linux.py --sglang-root /mnt/d/OpenKernelChallenge/serving/sglang --out NEW_RESULT_DIRECTORY [--entry ENTRY.py] [entry arguments].
Use a new output directory for each run. The launcher snapshots scripts, records installed packages and configures the isolated CUDA compiler and extracted Python headers. It does not install or modify system packages.

Important entries: kernel_runtime.py (default; --mode bench), validate_numeric_reference.py, validate_adaptive_selector.py, profile_marlin.py, measure_memory.py, validate_exported_candidate.py and rank_sweep.py. rank_sweep.py requires --candidate pointing to candidate_compact32_01. Exact commands and arguments are retained in each result; newer results include harness_snapshot.

Full-shard timing: --mode bench --hidden 4096 --intermediate 4096 --policies baseline adaptive_compact_32 --tokens 128 256 512 --samples 10 --repetitions 10 --replay-atol 0.0078125 --replay-relative-l2 0.0005

The notebook is an earlier structural study only. T4 cannot execute this BF16 Marlin path. No notebook upload is needed for the completed local work. Do not infer H200 performance from RTX3050 timings.

package_experiment.py refreshes the session evidence index and hashes using Windows Python; run it from this bundle. The initial structural packaging code and review are preserved separately.
