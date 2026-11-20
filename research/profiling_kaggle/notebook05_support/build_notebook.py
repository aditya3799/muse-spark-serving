import ast,hashlib,json,shutil
from pathlib import Path
import nbformat

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\OpenKernelChallenge\serving\profiling_kaggle')
core=(ROOT/'notebook04_support/timing_support.py').read_text(encoding='utf-8')
(HERE/'timing_support.py').write_text(core,encoding='utf-8',newline='\n')
helper=(ROOT/'notebook04_support/mapping_helper.py').read_text(encoding='utf-8')
(HERE/'mapping_helper.py').write_text(helper,encoding='utf-8',newline='\n')
slices=(HERE/'source_slices.py').read_text(encoding='utf-8')
support=(HERE/'source_study.py').read_text(encoding='utf-8')
manifest=json.loads((HERE/'source_manifest_05.json').read_text())
manifest.update(timing_support_sha256=hashlib.sha256(core.encode()).hexdigest(),study_sha256=hashlib.sha256(support.encode()).hexdigest())
(HERE/'source_manifest_05.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
nb=nbformat.v4.new_notebook(cells=[
md('''# 05 - Source patch draft and mapping reuse

Notebook 04 passed lifecycle tests and all 48 graph comparisons. Cached forward latency fell by a median 23.9%, but map updates cost more. This experiment moves from a handwritten module prototype to **source-selected statements from an isolated patch draft**, and directly measures whether refresh cost is amortized when a map is reused.

**Kaggle:** upload this file, select **GPU T4**, keep defaults and **Run All**. Only GPU 0 is used. No Internet, installs, model weights, datasets or credentials are required. All code is embedded. Keep the notebook private if you do not want to publish project source slices.

The draft is **not applied** to active V4. This notebook omits the full serving constructor, nonportable imports and real alignment/GEMMs. It does not load model weights or predict H200 speedup.

Download `v4_source_candidate_*.zip` at the end and put it in `profiling_kaggle/notebook05_results`. Run the export cell even if a section fails, then stop the GPU session.'''),
md('''## 1. What changed, and what the harness executes
The V4-relative draft adds a nonpersistent derived lookup buffer, an explicit in-place update/refresh API, and uses the lookup in the existing standalone mapping fallback. Static checks confirm the fused-mapping guard, complete alignment/GEMM tail and other existing methods are unchanged.

The harness selects the real map-initialization block and the pre-alignment prefix by AST from both source versions. It supplies only the attributes that prefix reads. The baseline mapping helper is copied exactly. These slices are neither the full serving class nor a stand-in GEMM benchmark.

The prefix includes hidden-state contiguity, branch selection, mapping and top-k score conversion. In the fused small-route branch it allocates uninitialized local IDs for a later kernel; **we never inspect their values or time that branch**. Fallback outputs are checked against an independent CPU reference.

The draft and full candidate source are saved locally beside this notebook's support files for review; no upload of those files is needed.'''),
code(core),code(slices),code(support),
code('SOURCE_MANIFEST='+repr(manifest)+'\nMAPPING_HELPER_SOURCE='+repr(helper)+'\n\nSAMPLES=11\nREPEATS=20\nTRIALS=3\nstudy=SourceStudy(SOURCE_MANIFEST,MAPPING_HELPER_SOURCE,samples=SAMPLES,repeats=REPEATS,trials=TRIALS)\nprint("Results directory:",study.out)'),
md('''## 2. Environment
Use within-run paired comparisons. Installed software and a clock snapshot are recorded; nothing is upgraded. CPU-only execution runs source checks but skips GPU timings.'''),
code('environment=study.environment()'),
md('''## 3. Source behavior and branch coverage
Check 672 configurations across empty, 1/8/9/32/128/256-token inputs, int32/int64, contiguous/strided IDs, four expert maps and six route patterns. Also test the no-map path and a noncontiguous map buffer.

On CUDA, eligible 1- and 8-token cases exercise the **deferred fused branch**. Only branch, shape/dtype and score outputs are checked there; actual alignment remains untested. Larger or ineligible cases validate exact mapping values. The output explicitly separates these counts.'''),
code('source_checks=study.source_checks()'),
md('''## 4. Exact source-prefix A/B timing
Compare baseline and candidate at 9/32/128/256 tokens, two integer dtypes and two route patterns, across three shuffled trials. Both use the same prefix statements except for the proposed lookup change.

Defaults: **48 cases, 192 timing rows and 2,112 raw samples**, plus **288 changed-input graph checks**. Each variant has eager and CUDA-graph execution, 11 samples and 20 repetitions per sample. No labels or profiler are active during timings. Setup, map refresh and correctness checks are excluded from this reuse-case measurement.

This prefix ends before real alignment; do not present its latency as full MoE or serving latency.'''),
code('benchmark=study.benchmark()'),
md('''## 5. Direct update-plus-reuse cycles
For 32 and 128 tokens, time **one mapping update followed by K prefix calls**, with K = **1, 8 or 64**. Candidate refresh is inside the same timed cycle. Baseline copies its map and runs the original prefix.

The incoming map changes during correctness checks. During timing, the workload still performs an update every cycle; repeated values do not skip the copy. This is a controlled reuse scenario, not an assertion about production update frequency.

Three trials, four execution variants and 11 samples produce **18 cases / 72 timing rows / 792 samples**. Each sample averages three complete cycles. Graph mode captures the whole update-plus-K sequence, not K separate graph replays. Intermediate results are computed normally; the last result is returned and checked.

Raw timings are **microseconds per cycle**. `comparisons.csv` also divides by K to show per-mapping cost including the update share. No separately measured medians are added. Combined notebook totals: **66 cases, 264 timing rows, 2,904 samples**.'''),
code('cadence=study.cadence()'),
md('''## 6. Separate trace
Trace six warmed source-prefix calls per variant at 128 tokens with int64 IDs and hot-local routing. The trace includes score conversion, unlike Notebook 04's narrower module wrapper. Only actual kernels/copies/memsets count as device operations; annotation spans are excluded. Compare within this trace, not by subtracting it from unprofiled latency.'''),
code('profiler=study.profile()'),
md('''## 7. Export
Return the ZIP containing source checks, case failures, raw samples, comparisons, direct reuse-cycle timings, trace evidence, provenance and checksums. The review will decide whether the source draft remains worthwhile under the stated update contract and will include the next runnable experiment.

Still untested: full model loading, actual alignment/GEMMs, multi-GPU communication, scheduler/tokenizer startup and H200 performance. Direct map mutations still need explicit refresh, and device/buffer replacement requires graph recapture.'''),
code('archive=study.export()\nfrom IPython.display import FileLink,display\ntry:\n    link=archive.relative_to(Path.cwd())\nexcept ValueError:\n    link=archive\ndisplay(FileLink(str(link)))'),
],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.10'}})
for i in (2,3,4):nb.cells[i].metadata['jupyter']={'source_hidden':True}
for cell in nb.cells:
    if cell.cell_type=='code':ast.parse(cell.source)
nbformat.validate(nb);nbformat.write(nb,HERE/'05_kaggle_source_candidate.ipynb')
print('Built',len(nb.cells),'cells;',sum(c.cell_type=='code' for c in nb.cells),'code cells')
