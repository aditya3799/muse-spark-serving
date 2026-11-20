import ast,hashlib,json
from pathlib import Path
import nbformat
HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\OpenKernelChallenge\serving');SOURCE=ROOT/'profiling_kaggle/notebook08_support'
files=['timing_support.py','source_slices.py','input_harness.py','ownership_study.py','update_study.py','mapping_helper.py']
for n in files:(HERE/n).write_text((SOURCE/n).read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
owner=ROOT/'sglang/python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py'
content=owner.read_text(encoding='utf-8')
audit={'source':str(owner.relative_to(ROOT)),'sha256':hashlib.sha256(content.encode()).hexdigest(),
    'exact_attribute_or_registration_lines':[{'line':i,'text':line.strip()} for i,line in enumerate(content.splitlines(),1) if 'self._expert_map' in line or 'register_buffer("_expert_map"' in line],
    'finding':'Constructor registration and forward read; no explicit runtime write found for this specific attribute in repository text search. Indirect buffer manipulation, device moves and framework integrations are not excluded.'}
(HERE/'SOURCE_AUDIT_09.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
manifest={'parent_source_manifest':json.loads((SOURCE/'source_manifest_08.json').read_text()),'source_audit':audit,
    'scope':'Unchanged source slices. Independent static mapping stages; graph-boundary diagnostic only.',
    'file_sha256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in files+['graph_study.py']}}
(HERE/'source_manifest_09.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
nb=nbformat.v4.new_notebook(cells=[
md('''# 09 - Static mapping across graph boundaries

Notebook 08 found that frequent-update overhead depends strongly on the Python update path. The checked-out serving class initializes its expert map and reads it in forward; explicit runtime mutation of that attribute was not found. That text audit cannot exclude indirect framework updates.

This experiment holds maps fixed and asks whether mapping savings persist when multiple source prefixes are captured together.

**Kaggle: upload this file, select GPU T4, and Run All.** Only GPU 0 is used. No installation, Internet, datasets, weights or credentials are needed. Keep the notebook private if you do not want to publish its embedded source slices.

Download `v4_graph_boundaries_*.zip` into `profiling_kaggle/notebook09_results`. Run export even after a section failure, then stop the GPU session.

**Scope:** runtime MoE preparation. This is not tokenizer/scheduler initialization, actual transformer layers, a complete model, or H200 validation. Active V4 and both drafts stay unchanged.'''),
md('''## 1. Same work, different execution boundaries

For baseline and single-owner mapping, execute 1, 8 or 32 independent source-prefix stages:

| Schedule | Execution |
|---|---|
| Eager | Python invokes every stage normally |
| Separate | Python replays one pre-captured graph per stage |
| Combined | One graph replay executes all stages |

Each stage has a distinct module and mapping buffer. Maps cycle through contiguous, shared and permuted assignments, then remain fixed. The stages share routing input values and do not consume each other's outputs. All outputs are retained and checked. There is no intervening attention, alignment, expert GEMM, communication or artificial filler computation.

The two graph schedules differ in graph boundaries and output allocations. This is a controlled graph-granularity experiment, not a faithful simulation of model layers or a measured production graph partition.'''),
*[code((HERE/n).read_text(encoding='utf-8')) for n in files[:5]],
code((HERE/'graph_study.py').read_text(encoding='utf-8')),
code('SOURCE_MANIFEST='+repr(manifest)+'\nMAPPING_HELPER_SOURCE='+repr((HERE/'mapping_helper.py').read_text(encoding='utf-8'))+'\n\nSAMPLES=11\nREPEATS=3\nTRIALS=3\nstudy=GraphStudy(SOURCE_MANIFEST,MAPPING_HELPER_SOURCE,samples=SAMPLES,repeats=REPEATS,trials=TRIALS)\nprint("Results directory:",study.out)'),
md('''## 2. Environment and correctness

Record the environment without changing installed packages. Re-run the inherited lifecycle gate: 192 configurations, 576 output checks, storage aliasing, moves, direct writes, graph visibility and checkpoint exclusion. All three prior implementations appear in that gate; only baseline and single-owner appear in the new timings.

CPU-only execution runs lifecycle checks and skips CUDA benchmarks. The inherited update-overhead class supplies only common timing utilities here; no update-overhead benchmark is invoked.'''),
code('environment=study.environment()\nlifecycle=study.lifecycle()'),
md('''## 3. Chain benchmarks

Token counts **9, 32, 128** with int64 IDs and top-k eight. Nine tokens is just above V4's small fused-mapping branch. That branch and its real kernel remain outside this experiment.

Three stage counts, three token counts and three shuffled trials produce **27 cases**. Six variants per case, eleven samples per variant and three complete chains per sample produce **162 timing rows and 1,782 raw samples**.

Before timing each case, change routing inputs twice, including negative sentinels, and check every stage output for all six variants. This gives **4,428 changed-input output checks** across the run. Pointer stability is checked; outputs are checked again after timing. Maps are never refreshed inside the timing interval.

Record CUDA-event time, synchronized wall time and host enqueue time per complete chain. Those intervals overlap; do not add or subtract them to assign CPU/GPU percentages. Comparisons also divide total time by stage count, which is a normalization rather than a separately measured stage latency. Input setup, capture, validation and profiler work are excluded from timing. Capture setup time is retained only as diagnostic metadata.'''),
code('benchmark=study.benchmark()'),
md('''## 4. Separate trace

Trace all six variants with eight stages and 128 tokens, three cycles each. Check whether the two graph schedules retain the same device work, and whether the candidate reduces per-stage operations. Only actual kernels/copies are counted. CPU-only capture or unavailable graph correlation is reported as partial.'''),
code('profiler=study.profile()'),
md('''## 5. Export

The ZIP includes all raw timing metrics, **135 matched comparisons**, lifecycle evidence, source audit/provenance, trace and checksums.

The result can tell us how capture boundaries affect this isolated mapping work. It cannot tell us the whole model's speedup, its graph partition, the production update frequency, or H200 performance. Framework buffer compatibility and the candidate's offset-pointer fused-kernel behavior remain unvalidated. Neither draft is promoted.'''),
code('archive=study.export()\nfrom IPython.display import FileLink,display\ntry:\n    link=archive.relative_to(Path.cwd())\nexcept ValueError:\n    link=archive\ndisplay(FileLink(str(link)))'),
],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.10'}})
for c in nb.cells:
    if c.cell_type=='code':ast.parse(c.source)
for c in nb.cells[2:8]:c.metadata['jupyter']={'source_hidden':True}
nbformat.validate(nb);nbformat.write(nb,HERE/'09_kaggle_graph_boundaries.ipynb')
print('Built',len(nb.cells),'cells')
