import ast
import hashlib
import json
from pathlib import Path
import nbformat

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\OpenKernelChallenge\serving')
previous=ROOT/'profiling_kaggle/notebook03_support'
source=(previous/'mapping_study.py').read_text(encoding='utf-8')
tree=ast.parse(source);lines=source.splitlines()
imports=[ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
functions=[ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('routes','reference','expert_mapping','write_csv','stats','parse_trace')]
klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MappingStudy')
methods=['\n'.join(lines[n.lineno-1:n.end_lineno]) for n in klass.body if isinstance(n,ast.FunctionDef) and n.name in ('environment','section','capture','measure','check','save')]
core='"""Timing utilities extracted from validated Notebook 03; no candidate code."""\n'+'\n'.join(imports)+'\n\n'+'\n\n'.join(functions)+'\n\nclass ExperimentCore:\n'+'\n\n'.join(methods)+'\n'
(HERE/'timing_support.py').write_text(core,encoding='utf-8')
extension=(HERE/'integration_study.py').read_text(encoding='utf-8')
helper_path=ROOT/'sglang/python/sglang/srt/layers/moe/token_dispatcher/standard.py'
helper_module=helper_path.read_text(encoding='utf-8')
node=next(n for n in ast.parse(helper_module).body if isinstance(n,ast.FunctionDef) and n.name=='_map_topk_ids_to_local_experts')
helper=ast.get_source_segment(helper_module,node)+'\n'
old=json.loads((previous/'source_manifest_03.json').read_text())
assert hashlib.sha256(helper.encode()).hexdigest()==old['mapping_helper_sha256']
assert hashlib.sha256((ROOT/'submission_v4.diff').read_bytes()).hexdigest()==old['patch_sha256']
owner=ROOT/'sglang/python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py'
manifest={'baseline':'V4 mapping helper and fallback ownership pattern; active serving code unchanged',
          'base_commit':old['base_commit'],'patch_sha256':old['patch_sha256'],
          'mapping_helper_sha256':old['mapping_helper_sha256'],
          'mapping_source_path':old['mapping_source_path'],
          'mapping_source_file_sha256':hashlib.sha256(helper_path.read_bytes()).hexdigest(),
          'owner_source_sha256':hashlib.sha256(owner.read_bytes()).hexdigest(),
          'timing_support_sha256':hashlib.sha256(core.encode()).hexdigest(),
          'integration_support_sha256':hashlib.sha256(extension.encode()).hexdigest(),
          'timing_origin_sha256':hashlib.sha256(source.encode()).hexdigest(),
          'note':'The notebook runs isolated nn.Module prototypes, not the full MuseSpark1xFp4Experts class. Original helper is exact; companion-buffer update API is experimental.'}
md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
nb=nbformat.v4.new_notebook(cells=[
md('''# 04 - Cached mapping: module lifecycle and integration experiment

Notebook 03 found consistent isolated T4 gains from a cached shifted lookup. Now we test whether that result survives **module ownership, map updates, device movement, and graph replay**.

**Run on Kaggle:** upload this file, select **GPU T4**, and **Run All**. Only GPU 0 is used. Internet can remain off; there are no package installs, credentials, datasets or model downloads. All code is embedded. Keep it private if you do not want to publish the source helper.

The experiment uses small `nn.Module` prototypes around the exact helper and the inspected fallback cast/contiguous steps. It does **not** instantiate the full serving class, load weights, execute alignment or change serving code. The existing eligible <=64-route fused path remains outside this experiment.

Download `v4_mapping_integration_*.zip` at the end and put it in `profiling_kaggle/notebook04_results`. Run the export cell even if a section fails or tracing is partial. Stop the GPU session after downloading.'''),
md('''## 1. Prototype and update contract
The baseline owns `_expert_map` as a nonpersistent buffer, matching the inspected serving class. The candidate also owns a nonpersistent `_expert_lookup` containing a leading `-1` followed by the original map. Forward uses the same int32 conversion and contiguous-output contract.

The candidate's `update_expert_map_` copies new map values and refreshes the existing lookup **in place**. Shapes, dtype and device must stay fixed. This preserves the addresses already read by a captured graph.

**Direct writes are not automatically detected.** Editing `_expert_map` directly requires `refresh_expert_lookup_` before using the cached path. A deliberate stale-cache test demonstrates this hazard and verifies the refresh remedy. Passing that test does not make unrefreshed writes safe.

Updates must be ordered before consumers on the same stream, or with an explicit event between streams. Moving/replacing buffers requires graph recapture. The experiment tests supported transitions; it does not exercise invalid old graphs after relocation.'''),
code(core),
code(extension),
code('SOURCE_MANIFEST = '+repr(manifest)+'\nMAPPING_HELPER_SOURCE = '+repr(helper)+'\n\nSAMPLES=11\nREPEATS=20\nTRIALS=3\nTOKENS=(9,32,128,256)\nstudy=IntegrationStudy(SOURCE_MANIFEST,MAPPING_HELPER_SOURCE,samples=SAMPLES,repeats=REPEATS,trials=TRIALS,tokens=TOKENS)\nprint("Results directory:",study.out)'),
md('''## 2. Environment
Record installed versions and GPU identity. No changes to the environment are attempted. CPU-only execution can check non-CUDA lifecycle behavior, but GPU timings and graph-specific tests are skipped.'''),
code('environment=study.environment()'),
md('''## 3. Lifecycle checks before timing

- **672 input configurations / 1,344 module outputs:** empty and small inputs, the 8/9-token boundary, larger inputs, int32/int64, contiguous/strided tensors, negative sentinels and varied mappings.
- No expert map: preserve the original passthrough cast behavior, including negative values.
- Reject wrong-shape or wrong-dtype updates before modifying the map.
- Demonstrate stale output after an unrefreshed direct write, then verify explicit refresh.
- Update mappings repeatedly while preserving both buffer addresses.
- Move CPU → GPU → CPU → GPU before capture; integer buffers must stay int32 even when the module's floating dtype is requested as float16.
- Exclude map/lookup buffers from `state_dict`; round-trip a small persistent marker to check checkpoint semantics. This is not a model checkpoint test.
- Replay graphs after changing inputs and mappings through the update API.
- Update from a producer stream, then consume after an explicit CUDA event.

Any unexpected lifecycle failure blocks the benchmark. Empty and <=8-token cases test only the mapping wrapper, not the existing fused aligner.'''),
code('lifecycle=study.lifecycle()'),
md('''## 4. Module forward benchmark
Compare **baseline module** and **cached module**, each under eager execution and CUDA graph replay. The same int32 cast and contiguous-output step are included in both.

Defaults: 16 configurations × three shuffled trials = **48 cases**, four timing variants per case = **192 timing rows / 2,112 raw samples**, with **288 changed-input graph checks** during the benchmark. Mapping initialization and updates happen outside forward timing. This measures the intended reuse case.

Each trial shuffles case order; each sample shuffles variant order. There are 11 samples, each averaging 20 calls. Correctness is checked before and after measurement. No profiler or labels are present in timed calls.

Graph replay changes dispatch and allocation behavior; CUDA-event intervals can include gaps. Compare variants within this run and inspect all trials. Do not convert these results into full serving or H200 percentages.'''),
code('benchmark=study.benchmark()'),
md('''## 5. Explicit update cost
Measure the update API separately. Baseline copies the map; candidate copies the map and refreshes its lookup. Each has eager and captured variants, over three trials.

This adds **12 timing rows / 132 samples**, for **204 total timing rows / 2,244 samples**. Do not add independently measured medians into an exact request-time estimate. Mapping lifetime and update frequency determine whether these costs matter.'''),
code('updates=study.update_cost()'),
md('''## 6. Separate kernel trace
After timing, profile six iterations of each module at 128 tokens, int64 inputs and hot-local routes. Compare actual kernel counts and work, excluding GPU annotation spans. Trace timings include instrumentation and must not be subtracted from benchmark latency.'''),
code('profiler=study.profile()'),
md('''## 7. Export results
Download the ZIP below or use Kaggle's output/file panel in `/kaggle/working`. It contains lifecycle results, failures/skips, timing samples, comparisons, update costs, trace evidence and checksums.

The next review will assess the actual mutation/migration contract before proposing a serving patch, and include the next runnable experiment. This notebook does not automatically promote the candidate.'''),
code('archive=study.export()\nfrom IPython.display import FileLink,display\ntry:\n    link=archive.relative_to(Path.cwd())\nexcept ValueError:\n    link=archive\ndisplay(FileLink(str(link)))'),
md('''Method references: [PyTorch module buffers and state dictionaries](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) and [CUDA graph storage/stream semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs).'''),
],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.10'}})
for index in (2,3):nb.cells[index].metadata['jupyter']={'source_hidden':True}
for cell in nb.cells:
    if cell.cell_type=='code':ast.parse(cell.source)
nbformat.validate(nb)
nbformat.write(nb,HERE/'04_kaggle_mapping_integration.ipynb')
(HERE/'source_manifest_04.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(HERE/'mapping_helper.py').write_text(helper,encoding='utf-8')
print('Built',len(nb.cells),'cells;',sum(c.cell_type=='code' for c in nb.cells),'code cells')
