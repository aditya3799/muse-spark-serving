import ast
import hashlib
import json
from pathlib import Path
import nbformat

HERE = Path(__file__).resolve().parent
ROOT = Path(r'D:\OpenKernelChallenge\serving')
source_path = 'python/sglang/srt/layers/moe/token_dispatcher/standard.py'
source = (ROOT / 'sglang' / source_path).read_text(encoding='utf-8')
tree = ast.parse(source)
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_map_topk_ids_to_local_experts')
helper = ast.get_source_segment(source, node) + '\n'
support = (HERE / 'routing_study.py').read_text(encoding='utf-8')
patch_hash = hashlib.sha256((ROOT / 'submission_v4.diff').read_bytes()).hexdigest()
assert patch_hash == '92054389830a8e5c07a37eb524810ac83690fbbcf88c60e3ae8a132909bb65f1'
provenance = {'candidate': 'V4', 'base_commit': '297e54fd105e974cdf401e5b4d20253a22c16f90',
              'patch_sha256': patch_hash, 'mapping_source_path': source_path,
              'mapping_source_file_sha256': hashlib.sha256((ROOT / 'sglang' / source_path).read_bytes()).hexdigest(),
              'mapping_helper_sha256': hashlib.sha256(helper.encode()).hexdigest(),
              'support_sha256': hashlib.sha256(support.encode()).hexdigest(),
              'note': 'Only the exact mapping helper is extracted from project source. The complete serving package is not imported or installed.'}
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
nb = nbformat.v4.new_notebook(cells=[
md('''# 02 - Routing overhead: eager execution versus CUDA graphs

Notebook 01 worked on T4. Its synthetic routing sequence took about 0.42 ms at both 8 and 128 tokens. This notebook tests whether fixed host/dispatch/allocation overhead and diagnostic labels help explain that behavior.

**Run:** upload this notebook to Kaggle, select **GPU T4**, and **Run All**. Only GPU 0 is used. No package installation, Internet access, dataset, model weights or credentials are needed. Keep it private if you do not want to publish the embedded project helper. Run the final export cell even if a section reports a failure, then download `v4_routing_overhead_*.zip`. Stop the interactive GPU session after downloading.

This notebook measures one exact project helper plus synthetic routing operations. It does **not** execute V4 Triton alignment, pack an aligned expert buffer, run expert GEMMs, measure startup, or predict H200 performance. The existing serving implementation may already use CUDA graphs.

Expected work: 60 eager correctness cases; 50 stage/shape/pattern comparisons with five changed-input graph checks each; 110 timing rows; one six-iteration trace. Runtime depends on the environment; progress is printed after each shape/pattern pair.'''),
md('''## 1. Embedded experiment code
Everything needed is included below. The mapping helper is extracted verbatim from our serving source, with hashes recorded for provenance. No files from Notebook 01 are required.'''),
code(support),
code('SOURCE_MANIFEST = ' + repr(provenance) + '\nMAPPING_HELPER_SOURCE = ' + repr(helper) + '\n\n# Defaults are intentionally bounded. Leave them unchanged for the first Kaggle run.\nSAMPLES = 15\nREPEATS = 20\nTOKENS = (1, 8, 9, 128, 256)\nstudy = RoutingStudy(SOURCE_MANIFEST, MAPPING_HELPER_SOURCE, samples=SAMPLES, repeats=REPEATS, tokens=TOKENS)\nprint("Results directory:", study.out)'),
md('''## 2. Environment
Record library versions and GPU identity/clock snapshot. A CPU-only session still runs correctness checks and exports a report, but supplies no substitute GPU timings. No compiler or Triton is required.'''),
code('environment = study.environment()'),
md('''## 3. Correctness before speed
An independent CPU reference checks mapping, stable route ordering, bucket counts and rounded padding counts. Cases include all-local, all-remote, explicit `-1` sentinels, uniform routing and concentrated local routing. A reversed local-expert map checks that results do not depend on a fixed mapping.

Inputs are flattened arrays of eight expert IDs per token. IDs are valid global indices or `-1`. The mapping helper operates elementwise on int32 IDs. Conversion from int64 is measured separately because that conversion belongs to this diagnostic input path, not to the helper itself.

Eight and nine tokens straddle 64 routes, but neither executes the actual V4 small-route kernel.'''),
code('correctness = study.correctness()'),
md('''## 4. Main experiment: timing without the profiler

| Stage | Work measured |
|---|---|
| `convert_i64_to_i32` | Convert the diagnostic input IDs |
| `original_mapping_i32` | Exact project helper: global IDs to this GPU's local IDs, preserving inactive routes |
| `sort_and_count` | Synthetic stable sorting and expert bucket counts |
| `padding_counts` | Round each count to a multiple of eight; no packed output buffer |
| `full_diagnostic` | All four stages together |

Each stage compares **eager** against **graph replay** with the same operator sequence and unchanged math. The full diagnostic also includes **eager_annotated_control**, which adds the labels used in Notebook 01, with no profiler active. This control measures label overhead within the same environment.

For every comparison: warm up on a side stream before capture; change routes and mappings in existing buffers and verify graph outputs; restore the benchmark input; warm all variants; randomize their order within each of 15 samples; time 20 repeated calls per sample; verify outputs again. Input copies, setup/capture, correctness checks, and profiling are outside the timer. Capture/setup time is reported separately in `benchmark_cases.csv`.

**Allocation distinction:** eager executes the allocating Python operators on every call; replay reuses captured graph-pool storage. Same operator sequence does not mean identical allocation cost. The difference combines host dispatch and allocation behavior; it is not a pure kernel-launch measurement. Reused-input measurements are warm microbenchmarks, not end-to-end request latency.

CUDA-event intervals include exposed gaps between operations. Synchronized wall times include host dispatch and synchronization. The small timing wrapper affects both, especially tiny stages. Do not sum isolated stage medians or turn them into serving percentages. Failed graph/correctness cases retain their error but publish no timing rows.'''),
code('benchmark = study.benchmark()'),
md('''## 5. Separate diagnostic trace
Only after all timing samples are complete, profile six eager iterations at 128 tokens with hot-local routing. The trace adds labels and instrumentation, so it is not a second latency benchmark.

`trace_device_operations.csv` counts only actual kernel/copy/memset events. GPU annotation ranges are excluded. Correlation IDs link GPU operations to their host launch and enclosing stage. `trace_stages.csv` preserves each iteration, so the first profiler iteration can be inspected separately. Missing GPU events or unmatched launches produce a partial status.'''),
code('profiler = study.profile()'),
md('''## 6. Export results
Download the ZIP below, or find it in Kaggle's output/file panel under `/kaggle/working`. Return the whole ZIP, preferably inside a new `notebook02_results` folder alongside the local notebook, to keep Notebook 01 outputs intact.

Files include `report.json`, `timings.csv`, `raw_samples.csv`, `benchmark_cases.csv`, the trace and trace CSVs when available, `mapping_helper.py`, `SUMMARY.txt`, `summary.html`, and integrity checksums. Failed/skipped sections are preserved.

We will first compare unannotated eager versus the labeled control, then eager versus replay, then inspect the isolated helper and trace. No serving change should be selected from a single synthetic speed ratio.'''),
code('archive = study.export()\nfrom IPython.display import FileLink, display\ntry:\n    link = archive.relative_to(Path.cwd())\nexcept ValueError:\n    link = archive\ndisplay(FileLink(str(link)))'),
md('''## Method references
The capture workflow follows [PyTorch CUDA graph documentation](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs): side-stream warmup, stable input/output addresses, changed inputs copied into existing buffers, and graph-private storage. Trace collection uses the [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html). Runtime package versions are recorded rather than upgraded.'''),
], metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.10'}, 'title': 'V4 routing overhead on Kaggle T4'})
nb.cells[2].metadata['jupyter'] = {'source_hidden': True}
for cell in nb.cells:
    if cell.cell_type == 'code':
        ast.parse(cell.source)
nbformat.validate(nb)
nbformat.write(nb, HERE / '02_kaggle_routing_overhead.ipynb')
(HERE / 'source_manifest_02.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
(HERE / 'mapping_helper.py').write_text(helper, encoding='utf-8')
print('Built', len(nb.cells), 'cells;', sum(c.cell_type == 'code' for c in nb.cells), 'code cells')
