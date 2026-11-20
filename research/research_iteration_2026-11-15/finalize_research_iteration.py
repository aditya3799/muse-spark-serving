from pathlib import Path
import hashlib,json,shutil,datetime
root=Path('/mnt/d/OpenKernelChallenge/serving')
out=root/'research_iteration_2026-11-15'
v1=out/'projection_candidate';v2=out/'projection_fp32_candidate'
receipt=json.loads((v2/'check_receipt.json').read_text())
validation=json.loads((v2/'gpu_validation.json').read_text())
assert validation['status']=='passed'
cases=[s for r in validation['rows'] for s in r['checks']]
summary=dict(status='passed',gpu_checks=len(cases),
    max_relative_l2=max(s['relative_l2'] for s in cases),
    max_abs=max(s['maxabs'] for s in cases),
    max_reference_l2=max(s['candidate_vs_fp32_l2'] for s in cases),
    job=receipt['id'],remote_status=receipt['status'],
    confirmed_serving_score=1.0111,target_achieved=False,
    patch_sha256=hashlib.sha256((v2/'candidate.diff').read_bytes()).hexdigest())
(out/'LATEST_STATUS.json').write_text(json.dumps(summary,indent=2))
old=json.loads((v1/'manifest.json').read_text())
old.update(status='Superseded: failed local default-BF16-reduction numerical screen. Previously submitted check remains diagnostic only.',superseded_by='projection_fp32_candidate',superseding_check=receipt['id'])
(v1/'manifest.json').write_text(json.dumps(old,indent=2))
(v1/'gpu_replay_checks.json').write_text(json.dumps(dict(passed=False,
    original_first_failure=dict(m=512,k=8192,n=4096,reduced=True,relative_l2=0.003904421115294099,max_abs=0.03125),
    threshold_relative_l2_lt=0.003,threshold_max_abs_le=0.03125,
    diagnosis='precision_diagnosis.json includes 24 follow-up measurements; original screening thresholds were not loosened.'),indent=2))
manifest=json.loads((v2/'manifest.json').read_text())
manifest.update(status='Local validation passed; H200 check pending',check_job=receipt['id'],check_status=receipt['status'])
(v2/'manifest.json').write_text(json.dumps(manifest,indent=2))
review=(out/'RESEARCH_REVIEW.txt').read_text()
review=review.replace('Iteration 2: shard replicated large projections - H200 CHECK QUEUED','Iteration 2: shard replicated large projections - FIRST VERSION SUPERSEDED')
review=review.replace('Added changed-input CUDA graph checks for noncontiguous inputs; see', 'Additional default-reduction graph checks FAILED for the first version; see')
review=review.replace('Fetch job82482c8c07c2422a.', 'Fetch corrected job32bcbb190b204dd6. Earlier job82482c8c07c2422a is diagnostic only.')
latest='''LATEST UPDATE: revised FP32-output projection candidate

H200 check 32bcbb190b204dd6 is queued; no new serving result yet.
Current candidate: projection_fp32_candidate/candidate.diff, 23881 bytes.
SHA256: 8b22315e0d4d6e71001cd7a85510a7abb35e228c1a98a93cf8e30399d8cbb791.
Confirmed serving score remains1.0111x. The1.5x objective is not achieved.

The first projection candidate (check82482c8c07c2422a) was superseded after
an extra GPU replay test with default BF16 reduction failed the predeclared
0.003 relative-L2 screen. It had already been submitted; do not promote it.
The failed screen and follow-up measurements are preserved, not relabeled.

Diagnosis: narrower BF16 GEMMs can select less accurate cuBLAS reductions.
FP32-reference tests isolated the dependence on PyTorch's reduction flag.
The revised helper requests torch.mm(...,out_dtype=torch.float32), converts
once to BF16 and gathers. It does not mutate global precision flags.
BF16 inputs, complete K reductions, original weights and final BF16 output
are retained. No donor kernel source was incorporated.

Revised validation:60 checks of the exact exported helper in CUDA graphs,
both projection geometries, M512/513/777/4096/16384, three changed-input seeds,
noncontiguous inputs and both BF16-reduction settings. All passed the SAME
0.003 relative-L2 /0.03125 max-absolute thresholds, plus a new requirement
that error against FP32 reference not exceed baseline error by more than1e-5.
Max relative L2 to baseline0.00215036; max absolute0.03125; max error relative
to FP32 reference0.00166283. Serial rank computations substitute for network
in these GPU checks. Earlier real2/4/8-process Gloo tests validate the unchanged
column gathering layout, not the revised CUDA math or an NCCL graph.

Revised timing:8 shape pairs, eight AB/BA samples, CUDA graphs; full projection,
identical full control, FP32-output local shard+BF16 cast, and layout conversion.
The saved compute exceeds local layout costs at every tested shape. Network
is absent, so these figures are NOT H200 or full-model speedup predictions.
Real H200 numerical, memory, NCCL/capture and throughput gates remain pending.
The 288MiB extra-live-storage projection estimate excludes workspaces/graph
retention. The local FP32 temporary is freed before gathering; measure full
deployment peak remotely rather than treating this estimate as a quota proof.

After the two checks, one check slot remains today; no scored slots remain.
Next:inspect32bcbb190b204dd6, reject if communication/memory erases benefit,
or use the remaining check at another load before considering scored promotion.
The active checkout/root patch remain V4 cached mapping. All new patches are
isolated exported candidates. No GPU process is intentionally left running.

Earlier research and experiment evidence follows:
------------------------------------------------
'''
(out/'RESEARCH_REVIEW.txt').write_text(latest+review)
sources=json.loads((out/'public_research_sources.json').read_text())
sources.extend([
 dict(name='PyTorch numerical accuracy',url='https://docs.pytorch.org/docs/2.15/notes/numerical_accuracy.html',finding='BF16 reduced-precision reduction is enabled by default and may truncate intermediate accumulations; this guided the controlled flag experiment.'),
 dict(name='PyTorch mm output dtype',url='https://docs.pytorch.org/docs/2.15/generated/torch.mm.html',finding='CUDA BF16 inputs support FP32 output. The revised helper uses this explicit output dtype and converts once to BF16.')])
(out/'public_research_sources.json').write_text(json.dumps(sources,indent=2))
(root/'NEXT_OPTIMIZATION.txt').write_text('''Latest research2026-11-15:confirmed scored result1.0111x;1.5x not achieved.
Corrected projection check32bcbb190b204dd6 queued at16 users.
Read research_iteration_2026-11-15/RESEARCH_REVIEW.txt and LATEST_STATUS.json.
Current artifact:research_iteration_2026-11-15/projection_fp32_candidate/candidate.diff.
Earlier projection check82482c8c07c2422a is superseded after a local numerical
screen failure; do not promote even if its H200 check passes.
Active sglang checkout/root patch.diff remain V4 cached mapping.
One daily check and zero scored slots remain after these submissions.
''')
root_manifest=json.loads((root/'candidate_manifest.json').read_text())
root_manifest['latest_scored_result']=dict(job='468701af35b84e07',passed=True,score=1.0111,
    source='research_iteration_2026-11-15/scored_468701af35b84e07.json')
root_manifest['latest_research_check']=summary
(root/'candidate_manifest.json').write_text(json.dumps(root_manifest,indent=2))
readme=root/'optimization_marlin_tiles_2026-11-15/README.txt'
body=readme.read_text().split('\n',1)[1]
readme.write_text('LATEST: scored job468701af35b84e07 passed at1.0111x. See SUBMISSION_STATUS.txt. New research is in ../research_iteration_2026-11-15. Historical profiling snapshot follows.\n'+body)
for name in ('diagnose_projection_precision.py','projection_replay_checks.py','finalize_research_iteration.py'):
    shutil.copy2(Path('/mnt/c/Windows/Temp')/name,out/name)
checks={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='research_checksums.json'}
(out/'research_checksums.json').write_text(json.dumps(checks,indent=2))
assert hashlib.sha256((root/'patch.diff').read_bytes()).hexdigest()=='cf02617f790e28c9229ac9aec5e517f04344c49491347ef732a352fbaa3aa4a4'
print(json.dumps(summary,indent=2))
