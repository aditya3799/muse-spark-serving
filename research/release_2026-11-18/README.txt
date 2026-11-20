TODAY'S NEW CHECK CANDIDATE

Use candidate.diff with --moe-runner-backend humming.
SHA256: 8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b
Base: 297e54fd105e974cdf401e5b4d20253a22c16f90

This is the proven Humming build plus a guarded extension of adaptive expert
partitioning to larger prefills. Only two files differ from the proven parent.
Unsupported configurations fall back, and the existing <=4,096-route path stays
unchanged. All computations retain the original precision and canonical weights.

Validation passed: clean-base patch application, exact exported source hashes,
Python3.12 syntax for all10 files,58 native-configuration CPU checks,35 selector
checks,34 integrated named CPU checks (including140 wrapper cases), and32 local
GPU cases with56 bitwise raw/final comparisons. The GPU test exercised actual new
preparation and partition execution at H=I256/E32/M513,768. It was not a full model
or H200 test and measured no serving speedup. The fixture's44MiB tensor peak is not
whole-model total-device-memory compliance.

NEXT: use one16-user unscored target check. CHECK_COMMAND.txt has the prepared
command; check_request.json and run_request.json contain exact request payloads.
Neither has been sent. Review startup, correctness, later turns, memory, TTFT and
indicative performance before deciding whether to spend the scored run.

The new candidate's speedup is unknown. The proven fallback remains1.2237x and
is preserved in proven/candidate.diff. That parent is already submitted; copying
it into this folder does not create a new gain or justify a duplicate submission.

The earlier checks fa0d2a0fcf584339 and6570fbd9e1b64ce4 are now done and passed
correctness/later-turn checks. Their16-user indicative ratios are1.069x and1.052x;
these are not full serving scores. Fresh response receipts are in evidence/.

All active serving files and root patch.diff were preserved. No TTS/service action,
new remote check, scored submission or paid compute occurred during preparation.
See RELEASE_MANIFEST.json for exact qualification limits and evidence paths.
