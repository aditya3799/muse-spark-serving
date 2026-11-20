"""Consolidate completed evidence without modifying the submitted serving source."""

import contextlib
import hashlib
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from summarize_benchmark import summarize
from tile_study import manifest


def main():
    root = Path(__file__).resolve().parent
    serving = Path(r"D:\OpenKernelChallenge\serving")
    current = json.loads((serving / "candidate_manifest.json").read_text())
    preserved = (
        hashlib.sha256((serving / "patch.diff").read_bytes()).hexdigest()
        == current["sha256"]
    )
    preserved &= all(
        hashlib.sha256((serving / "sglang" / p).read_bytes()).hexdigest() == h
        for p, h in current["source_file_sha256"].items()
    )
    assert preserved
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "-v", "test_experiment"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    (root / "unit_tests.txt").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    assert tests.returncode == 0
    benches = []
    for path in sorted(root.rglob("benchmark.json")):
        report = json.loads(path.read_text())
        if report.get("timing_rows"):
            with contextlib.redirect_stdout(io.StringIO()):
                summarize(path)
            analysis = json.loads(path.with_name("analysis.json").read_text())
            benches.append(
                dict(
                    path=str(path.relative_to(root)),
                    status=report["status"],
                    arguments=report["arguments"],
                    checks=analysis["checks"],
                    max_abs=analysis["max_abs"],
                    groups=analysis["groups"],
                )
            )
            manifest(path.parent)
    serial_path = root / "rank_sweep_compact32/serial_rank_summary.json"
    serial = (
        json.loads(serial_path.read_text())
        if serial_path.exists()
        else {"status": "incomplete"}
    )
    candidate_path = root / "candidate_compact32_01/candidate_manifest.json"
    candidate = json.loads(candidate_path.read_text())
    memory_path = root / "memory_full_01/memory.json"
    memory = (
        json.loads(memory_path.read_text())
        if memory_path.exists()
        else {"status": "pending"}
    )
    validations = []
    for name in (
        "alignment_validation.json",
        "numeric_reference.json",
        "selector_validation.json",
        "repeatability.json",
    ):
        for path in sorted(root.rglob(name)):
            value = json.loads(path.read_text())
            validations.append(
                {
                    "path": str(path.relative_to(root)),
                    **{
                        k: v
                        for k, v in value.items()
                        if k not in ("rows", "checks", "runtime")
                    },
                }
            )
    plan = json.loads((root / "SESSION_PLAN.json").read_text())
    end = datetime.now(timezone.utc)
    start = datetime.fromisoformat(plan["started_utc"].replace("Z", "+00:00"))
    review = dict(
        status="local profiling complete; target serving result pending",
        started_utc=plan["started_utc"],
        reviewed_utc=end.isoformat(),
        elapsed_minutes=(end - start).total_seconds() / 60,
        target_serving_1_5x_established=False,
        submitted_candidate_preserved=preserved,
        submitted_patch_sha256=current["sha256"],
        new_judge_runs_launched=0,
        existing_job=current["unscored_job"],
        unit_tests_passed=13,
        selected_experimental_candidate=candidate,
        serial_rank_proxy=serial,
        memory=memory,
        validation_runs=validations,
        benchmark_runs=benches,
        rejected_approaches=[
            "Global expert-density tile selection: large skew regressions",
            "Sorted EP specialization: no meaningful local gain",
            "16-row compact selection as universal policy: slow-rank fallback risk",
        ],
        limitations=[
            "RTX3050 SM86, synthetic weights/routes, source-minimal SGLang runtime",
            "Ratios are relative to V4 cached mapping, not stock serving",
            "Serial virtual ranks reuse synthetic local weight values; no simultaneous eight-GPU execution",
            "Dense-reference and tolerance screens do not prove greedy-token agreement",
            "Full-model memory, TTFT, and H200 serving score remain unmeasured",
        ],
    )
    (root / "SESSION_RESULTS.json").write_text(
        json.dumps(review, indent=2), encoding="utf-8"
    )
    candidate["local_component_validation"] = dict(
        serial_rank_status=serial["status"],
        exact_exported_method_executed=True,
        selector_cases=192,
        dense_reference_cases=80,
        no_serving_validation=True,
    )
    candidate_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    lines = [
        "PROFILING SESSION REVIEW - 2026-11-15",
        "",
        "No 1.5x end-to-end serving claim is established. The submitted V4 cached-mapping patch remains unchanged; the existing H200 check is still pending.",
        "",
        f"Selected experimental patch: candidate_compact32_01/candidate_unsubmitted.diff ({candidate['patch_bytes']} bytes)",
        f"SHA256: {candidate['patch_sha256']}",
        "This is a cumulative patch against 297e54fd, verified by applying it in an isolated Git root and comparing every output source. It is prepared for H200 validation, not integrated or submitted.",
        "",
        "The patch preserves original Marlin arithmetic and selects a compact 32-row route table on the GPU only when each active local expert fits in one tile. Otherwise it retains the baseline tile. Exactly one projection performs active work. Small workloads and unsupported selector sizes retain the original path.",
        "",
        "Measured serial max-rank GRAPH proxy (baseline max / candidate max):",
    ]
    for row in serial.get("rows", []):
        if row["pattern"] != "inactive":
            lines.append(
                f"  M={row['tokens']:4d} {row['pattern']:11s} {row['serial_max_rank_ratio']:.4f}x"
            )
    lines += [
        "",
        "These are sequential synthetic expert-partition measurements on one RTX3050, not H200 timings, concurrent distributed execution, stock throughput, or a serving score. Inactive work is excluded from useful-work summaries. A single local 1.60x point from the earlier 16-row policy is not the final result.",
        "",
        "What was established:",
        "- Real CUDA/Triton alignment, both Marlin GEMMs, activation and finalization ran locally at 4096-wide expert dimensions.",
        "- The local profile put about 99.5% of kernel time in the two GEMMs.",
        "- Fixed small tiles can regress concentrated routes by roughly 3x; that blanket policy was rejected.",
        "- Sorted-EP compiled and passed local checks but was essentially neutral.",
        "- Compact selection preserves every active route; changed-input graph tests cover balanced, concentrated, remote and inactive routing.",
        "- Independent dense-reference checks also cover duplicate slots and noncontiguous routing inputs.",
        "- Unchanged baseline execution is not always bitwise repeatable. Original strict failures and repeatability diagnostics are retained. Later replay screens use atol=0.0078125 and relative L2<=0.0005; candidate-vs-baseline uses atol/rtol=0.01/0.01. These are component screens, not token-agreement certification.",
        "",
        "Memory:",
    ]
    if memory.get("status") == "passed":
        for tokens in sorted({r["tokens"] for r in memory["rows"]}):
            selected = {r["policy"]: r for r in memory["rows"] if r["tokens"] == tokens}
            extra = (
                selected["adaptive_compact_32"]["graph_peak_above_inputs_bytes"]
                - selected["baseline"]["graph_peak_above_inputs_bytes"]
            )
            lines.append(
                f"  M={tokens}: extra local-layer graph peak {extra / 1048576:.3f} MiB"
            )
    else:
        lines.append("  Standalone local-layer graph memory measurement not completed.")
    lines += [
        "This does not establish the full-model 2GB/GPU limit; graph sharing, other layers, communication and KV cache are absent.",
        "",
        "Next: obtain the queued H200 result, then profile/test this candidate on H200 with real routing and full-model correctness, TTFT and peak memory. Do not spend a scored submission based only on these RTX results. Do not revive discarded V5/V6 or the closed mapping-notebook series.",
        "",
        "Reproduction: use run_local_linux.py with the isolated WSL venv /home/aditya/.venvs/okc-marlin-tiles. Each newer run retains harness_snapshot and package/compiler metadata. rank_sweep_compact32/exported_candidate_provenance.json identifies the exact candidate source executed. Raw samples, failures, profiler trace, summaries and source hashes are preserved.",
    ]
    text = "\n".join(lines) + "\n"
    (root / "SESSION_REVIEW.txt").write_text(text, encoding="utf-8")
    old = root / "LOCAL_REVIEW.txt"
    if not (root / "STRUCTURE_REVIEW_INITIAL.txt").exists():
        (root / "STRUCTURE_REVIEW_INITIAL.txt").write_bytes(old.read_bytes())
    old.write_text(text, encoding="utf-8")
    readme = root / "README.txt"
    if not (root / "README_STRUCTURE_INITIAL.txt").exists():
        (root / "README_STRUCTURE_INITIAL.txt").write_bytes(readme.read_bytes())
    readme.write_text(
        "MARLIN EXPERIMENT BUNDLE\n\nStart with SESSION_REVIEW.txt. SESSION_RESULTS.json records the full evidence index.\nThe selected unapplied cumulative patch is candidate_compact32_01/candidate_unsubmitted.diff.\n\nRuntime: WSL Ubuntu, /home/aditya/.venvs/okc-marlin-tiles/bin/python.\nLauncher: run_local_linux.py --sglang-root /mnt/d/OpenKernelChallenge/serving/sglang --out NEW_RESULT_DIRECTORY [--entry ENTRY.py] [entry arguments].\nUse a new output directory for each run. The launcher snapshots scripts, records installed packages and configures the isolated CUDA compiler and extracted Python headers. It does not install or modify system packages.\n\nImportant entries: kernel_runtime.py (default; --mode bench), validate_numeric_reference.py, validate_adaptive_selector.py, profile_marlin.py, measure_memory.py, validate_exported_candidate.py and rank_sweep.py. rank_sweep.py requires --candidate pointing to candidate_compact32_01. Exact commands and arguments are retained in each result; newer results include harness_snapshot.\n\nFull-shard timing: --mode bench --hidden 4096 --intermediate 4096 --policies baseline adaptive_compact_32 --tokens 128 256 512 --samples 10 --repetitions 10 --replay-atol 0.0078125 --replay-relative-l2 0.0005\n\nThe notebook is an earlier structural study only. T4 cannot execute this BF16 Marlin path. No notebook upload is needed for the completed local work. Do not infer H200 performance from RTX3050 timings.\n\npackage_experiment.py refreshes the session evidence index and hashes using Windows Python; run it from this bundle. The initial structural packaging code and review are preserved separately.\n",
        encoding="utf-8",
    )
    package = root / "package_experiment.py"
    if not (root / "package_structure_initial.py").exists():
        (root / "package_structure_initial.py").write_bytes(package.read_bytes())
    package.write_text(
        '"""Refresh the profiling session evidence and checksum bundle."""\nfrom finalize_session import main\n\nif __name__ == "__main__":\n    main()\n',
        encoding="utf-8",
    )
    (root / "RESUME.txt").write_text(
        "Read SESSION_REVIEW.txt and SESSION_RESULTS.json. The current draft is candidate_compact32_01/candidate_unsubmitted.diff. Active V4 cached-mapping submission is unchanged; existing H200 check f1513498e8d84de8 is pending. No new judge run was launched. Full-model 1.5x is not established.\n",
        encoding="utf-8",
    )
    (root / "experiment_manifest.json").write_text(
        json.dumps(
            {
                k: v
                for k, v in review.items()
                if k not in ("benchmark_runs", "validation_runs", "memory")
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "NEXT_OPTIMIZATION.updated.txt").write_text(
        "Profiling session 2026-11-15: read optimization_marlin_tiles_2026-11-15/SESSION_REVIEW.txt. Tested experimental compact 32-row Marlin EP selection is saved under candidate_compact32_01/candidate_unsubmitted.diff. Original submitted V4 cached mapping is unchanged. H200 check f1513498e8d84de8 remains pending; no new submission. RTX component gains do not establish 1.5x serving. Next step is H200 validation with real routes, greedy-token agreement, TTFT and peak memory.\n",
        encoding="utf-8",
    )
    for directory in (root / "candidate_compact32_01", root / "rank_sweep_compact32"):
        manifest(directory)
    checksums = {
        str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
        and not any(x in p.parts for x in ("__pycache__", ".ruff_cache", ".git"))
        and p != root / "bundle_checksums.json"
    }
    (root / "bundle_checksums.json").write_text(
        json.dumps(checksums, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                files=len(checksums),
                elapsed_minutes=review["elapsed_minutes"],
                serial_rank_status=serial["status"],
                memory_status=memory["status"],
                submission_preserved=preserved,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
