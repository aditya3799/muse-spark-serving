from pathlib import Path
import hashlib,json,shutil,datetime
root=Path('/mnt/d/OpenKernelChallenge/serving')
out=root/'research_iteration_2026-11-15'
candidate=out/'projection_candidate'
result=json.loads((out/'scored_468701af35b84e07.json').read_text())
bench=json.loads((out/'prefill_tile_sweep/benchmark.json').read_text())
rows=[{k:r[k] for k in ('tokens','pattern','policy','median_paired_ratio')} for r in bench['timing_rows'] if r['schedule']!='eager']
(out/'prefill_tile_summary.json').write_text(json.dumps(rows,indent=2))
sources=[
 ('SGLang roadmap','https://github.com/sgl-project/sglang/issues/44678','Router CuTe GEMV already in pinned source; MegaMoE and megakernel directions are not drop-in Marlin optimizations.'),
 ('SGLang cookbook','https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Meta/MuseSparkOpen.mdx','H200 uses Marlin; current sequence-parallel NVFP4 prefill path requires Blackwell. Local forward_dispatched explicitly rejects Marlin.'),
 ('Meta megakernel','https://github.com/facebookresearch/sgl-muse-mega','GB300/CUDA13.1 implementation; useful launch/fusion ideas but cannot run on H200 as supplied.'),
 ('FP8 speculative regression','https://github.com/sgl-project/sglang/issues/32938','Kimi MLA FA3 verification materialized full cache in BF16. Our Muse MHA path passes FP8 paged KV directly; do not attribute this bug to our run.'),
 ('Marlin EP tile issue','https://github.com/vllm-project/vllm/issues/48066','Reported occupancy cost from large EP tiles. Our earlier adaptive approach addressed this locally; the new larger-M sweep rejects blanket smaller prefill tiles.'),
 ('Marlin original','https://github.com/IST-DASLab/marlin','Weight-only kernel design and occupancy/memory overlap background; not a serving-speedup claim.'),
 ('PyTorch CUDA graphs','https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html','Graphs address launch overhead. Startup already confirms prefill and verify capture; torch.compile is not an unmeasured automatic improvement.'),
 ('PyTorch profiler','https://docs.pytorch.org/docs/stable/profiler','Separate profiler runs from latency measurements; record_shapes/stack tracing can perturb execution.'),
 ('PyTorch tensor parallel','https://docs.pytorch.org/docs/main/distributed/tensor.parallel.html','Columnwise linear sharding motivates computing independent output columns and gathering them.'),
 ('NCCL CUDA graphs','https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2262/user-guide/docs/usage/cudagraph.html','Collectives support capture, with uniform capture participation across ranks. Actual SGLang multi-GPU integration still needs the remote check.'),
 ('H200 public kernels','https://github.com/alygg77/okc-muse-spark-kernels','Reviewed H200 measured facts, negative results and prefill design; no donor source incorporated in the projection candidate.'),
 ('Additional public kernels','https://github.com/Kushagrabainsla/muse-spark-open','Reviewed README descriptions of persistent MoE and fused attention; component-only claims, no transplant performed.'),
]
(out/'public_research_sources.json').write_text(json.dumps([dict(name=a,url=b,finding=c) for a,b,c in sources],indent=2))
text='''RESEARCH ITERATION - 2026-11-15

Outcome so far
The scored compact32 candidate passed at 1.0111x stock, not the requested 1.5x.
Per-load ratios: 1.0227, 1.0049, 1.0084, 1.0084. Correctness, later-turn
rescoring and memory checks passed. Full JSON and startup log are preserved here.
There is no per-kernel H200 profiler trace in the returned result. Startup logs
do not prove which conditional kernel branch executed on measured requests.

Iteration 1: larger-batch MoE tile experiment - REJECTED
Executed actual SGLang NVFP4 Marlin GEMMs, alignment, SiLU and finalization on
the RTX 3050, H=I=4096, 32 local/256 global experts, topk8, M=1024/2048/4096.
Paired AB/BA eager and CUDA graph timings; uniform and hot-rank routing; six
samples each. Correctness/replay screens passed. A/A graph controls 0.9997-1.0035x.
Fixed32 graph ratios 0.601-0.795x. Fixed48 ratios 0.770-1.118x; its isolated
M1024 uniform win regresses on concentrated routes and larger batches.
Keep existing tile behavior. These are component timings, not H200 estimates.

Iteration 2: shard replicated large projections - H200 CHECK QUEUED
Pinned Muse code computes large pooler/unpooler BF16 projections redundantly
on all TP ranks on H200. The candidate computes complete output-column slices
on each rank and uses existing GroupCoordinator.all_gather to reconstruct the
full tensor. It does not split the K reduction, repack weights, lower precision,
or remove tokens. It retains stock paths for small M, non-Hopper GPUs, SP,
attention DP/CP, deterministic inference and unsupported layouts.
Eligibility is generic geometry (M>=512, aligned BF16 dense weights), not prompt
contents or contest load labels. This is a performance hypothesis, not a claim.

Built cumulative candidate on the exact scored compact32 artifact, with one
new helper and two projection call sites. Patch is 23,714 bytes, six files,
applies exactly to pinned base and matches exported source hashes. Active
checkout and root patch.diff remain the earlier V4 cached-mapping baseline.

Validation:
- Actual Gloo collectives at 2, 4 and 8 processes, six cases per world on every
  rank: bitwise output equality. Exact exported helper/source-extracted SGLang
  gather, with F.linear dispatcher binding. This is NOT an NCCL graph test.
- Twelve local GPU projection cases, M511 through16384, both8192->4096 and
  4096->8192: max relative L2 about0.000205, maxabs0.015625. Single-GPU compute
  and layout timings separately recorded; no communication speedup inferred.
- Added changed-input CUDA graph checks for noncontiguous inputs; see
  projection_candidate/gpu_replay_checks.json for status and exact results.
- Extra live tensor storage bound at M16384,N8192,TP8 is288MiB per projection,
  excluding cuBLAS/NCCL workspaces and graph pool retention. Full-model peak
  memory and launch/collective behavior require the judge.

Check job: 82482c8c07c2422a, 16 users, judge-default flags. Queued at submission.
No new scored run: today's single scored slot was already consumed. Two daily
check slots remain after this submission (subject to live status).

Iteration 3: persistent MoE port feasibility - NOT READY
The protected v139 donor is214000bytes (SHA A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09),
so file size alone is not the blocker. Its prepare() retains base/fat/unit/WGS
layouts for each of tiny/larger formats. With E32,H=I4096, codes+FP8 scales are
0.84375GiB per layout, eight layouts total6.75GiB per layer before scratch.
Stock Marlin codes+BF16 scales are roughly0.9375GiB. A direct whole-model port
would violate the2GB/GPU allowance by far. It also fixes local expert offset0,
needs global-scale/norm adaptation and M>384 handling. Do not import prepare()
or claim the standalone2.39x as a serving gain. A credible port must use one
canonical weight layout across decode/prefill and prove all-rank semantics.

Next decision
Fetch job82482c8c07c2422a. Require correctness, later-turn validation where
available, <=2GB additional memory, TTFT limit and a convincing serving gain.
If projection communication erases compute savings, reject this branch rather
than stack speculative changes on it. If it wins, run another load check before
using a scored slot after reset. A full H200 profile or validated single-layout
MoE port is still needed to make a credible1.5x proposal.

Research URLs and applicability decisions: public_research_sources.json.
Public donor files were read for research only; none were copied into this patch.
'''
(out/'RESEARCH_REVIEW.txt').write_text(text)
next_text='''Latest state: 2026-11-15 iterative research
Scored job468701af35b84e07 passed at1.0111x;1.5x objective not achieved.
New projection candidate check82482c8c07c2422a queued at16 users.
Read research_iteration_2026-11-15/RESEARCH_REVIEW.txt and check_receipt.json.
Candidate: research_iteration_2026-11-15/projection_candidate/candidate.diff.
Active sglang checkout/root patch.diff remain V4 cached mapping; do not assume
they contain either submitted compact32 or the new projection experiment.
No further scored submissions until H200 evidence supports promotion.
'''
(root/'NEXT_OPTIMIZATION.txt').write_text(next_text)
receipt=json.loads((candidate/'check_receipt.json').read_text())
manifest=json.loads((candidate/'manifest.json').read_text())
manifest.update(status='H200 check submitted; result pending',check_job=receipt['id'],check_status=receipt['status'])
(candidate/'manifest.json').write_text(json.dumps(manifest,indent=2))
for script in ('projection_replay_checks.py','save_research_iteration.py'):
    shutil.copy2(Path('/mnt/c/Windows/Temp')/script,candidate/script)
checks={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='research_checksums.json'}
(out/'research_checksums.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(dict(files=len(checks),review=str(out/'RESEARCH_REVIEW.txt'),job=receipt['id'])))
