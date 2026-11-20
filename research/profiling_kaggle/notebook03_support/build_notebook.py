import ast
import hashlib
import json
from pathlib import Path
import nbformat

HERE = Path(__file__).resolve().parent
ROOT = Path(r'D:\OpenKernelChallenge\serving')
helper_path = ROOT / 'sglang/python/sglang/srt/layers/moe/token_dispatcher/standard.py'
source = helper_path.read_text(encoding='utf-8')
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name=='_map_topk_ids_to_local_experts')
helper = ast.get_source_segment(source, node)+'\n'
support = (HERE/'mapping_study.py').read_text(encoding='utf-8')
patch_hash = hashlib.sha256((ROOT/'submission_v4.diff').read_bytes()).hexdigest()
assert patch_hash=='92054389830a8e5c07a37eb524810ac83690fbbcf88c60e3ae8a132909bb65f1'
previous = json.loads((ROOT/'profiling_kaggle/notebook02_support/source_manifest_02.json').read_text())
assert hashlib.sha256(helper.encode()).hexdigest()==previous['mapping_helper_sha256']
manifest = {'baseline': 'V4 fallback helper; serving code unchanged', 'base_commit': previous['base_commit'],
            'patch_sha256': patch_hash, 'mapping_source_path': previous['mapping_source_path'],
            'mapping_source_file_sha256': hashlib.sha256(helper_path.read_bytes()).hexdigest(),
            'mapping_helper_sha256': hashlib.sha256(helper.encode()).hexdigest(),
            'support_sha256': hashlib.sha256(support.encode()).hexdigest(),
            'candidate': 'Shifted lookup with clamp_min(-1), implemented only inside this experiment.'}
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
nb = nbformat.v4.new_notebook(cells=[
md('''# 03 - One candidate for the expert-ID mapping fallback

Notebook 02 showed that host dispatch, allocations and labels materially affect the synthetic diagnostic. This experiment tests **one actual mapping change**, using the exact project helper as baseline. V4 already fuses mapping into eligible inputs of up to 64 routes; this notebook focuses on **9, 32, 128 and 256 tokens at top-k 8**, where the standalone fallback is relevant.

**Kaggle:** upload this file, select **GPU T4**, and **Run All**. Only GPU 0 is used. No Internet, package installation, model download, dataset or credentials are needed. Keep the defaults for the first run. This notebook is self-contained and does not require Notebook 01 or 02 files. Keep it private if you do not want to publish the embedded source helper.

Download the final `v4_mapping_candidate_*.zip` and place it locally in `profiling_kaggle/notebook03_results`. Run the export cell even if a section reports failure or a partial trace, then stop the interactive GPU session.

This is an isolated experiment. It does not change serving code, widen the fused-kernel threshold, measure startup, or predict H200 latency.'''),
md('''## 1. Candidate and its assumptions

The baseline separately detects negative IDs, clamps indices for safe lookup, and selects the `-1` result for dropped routes.

The candidate prepends one entry to the mapping table:

```python
lookup = torch.cat((mapping.new_full((1,), -1), mapping))
result = lookup[topk_ids.clamp_min(-1) + 1]
```

Every negative ID becomes index 0, whose value is `-1`. Every valid nonnegative ID `g` becomes `g + 1`, which stores the original `mapping[g]`. The clamp happens **before** adding one, including for the smallest int32 negative value. Nonnegative IDs must remain valid indices, as in the baseline.

We test both **cached lookup** (prepared once) and **rebuilt lookup** (prepared on every call). Cached storage adds **1,028 bytes per int32 mapping table**. When the original mapping changes, this lookup must be refreshed before reuse. For captured graphs, refresh the existing storage rather than replacing its pointer.

The exact helper is embedded below. Other code implements only the experiment; the full serving package is neither imported nor installed.'''),
code(support),
code('SOURCE_MANIFEST = '+repr(manifest)+'\nMAPPING_HELPER_SOURCE = '+repr(helper)+'\n\nSAMPLES = 11\nREPEATS = 20\nTRIALS = 3\nTOKENS = (9, 32, 128, 256)\nstudy = MappingStudy(SOURCE_MANIFEST, MAPPING_HELPER_SOURCE, samples=SAMPLES, repeats=REPEATS, trials=TRIALS, tokens=TOKENS)\nprint("Results directory:", study.out)'),
md('''## 2. Record the environment
We compare variants **within this run**. Notebook 02 used different library versions from Notebook 01, so cross-notebook differences are not evidence of a code improvement. No upgrades are attempted. A CPU-only session runs correctness checks but skips GPU timings.'''),
code('environment = study.environment()'),
md('''## 3. Correctness gate
Check **672 input cases / 4,032 outputs** against an independent CPU reference. Coverage includes empty inputs; the 8/9-token boundary; int32 and int64; strided and contiguous tensors; dropped, all-local, all-remote and uneven routes; and contiguous, permuted, shared-expert and all-remote mapping tables. Negative values include `-1`, `-2` and the int32 minimum.

Both candidate variants and the baseline must preserve values, output shape/dtype/device/contiguity, and input contents. We test the helper alone and the surrounding fallback cast/contiguous operations. Input IDs fit int32; this does not establish that arbitrary int64 values can safely be narrowed.'''),
code('correctness = study.correctness()'),
md('''## 4. Repeated A/B measurements

| Scope | What is included |
|---|---|
| `helper_only` | Mapping on int32 or int64 IDs, without an added cast |
| `fallback_cast_map` | The serving fallback's int32 cast, mapping, and contiguous-output step |

Each scope measures baseline, cached lookup and rebuilt lookup in both **eager** and **CUDA graph** execution. Defaults give 32 configurations per trial, three trials, and six variants per configuration: **96 benchmark cases, 576 timing rows, 6,336 raw samples**. There are 1,152 changed-input graph checks before timing, plus checks after timing. Progress prints every eight cases.

Case order is shuffled separately for each trial; variant order is shuffled within each sample. Each sample averages 20 calls, and each variant has 11 samples per trial. Capture uses side-stream warmup and retained input/output storage. Inputs and mappings are changed in place and checked before benchmark inputs are restored.

No profiling labels or active profiler are present in timing calls. Input generation/copies, graph capture and correctness checks are outside the timer. Cached lookup preparation is excluded here and measured next; rebuilt lookup includes preparation on each invocation.

**Interpretation:** graphs change host dispatch and allocator behavior together. Event intervals can include gaps between kernels; they are not pure kernel time. Compare like-for-like execution modes and inspect all three trials. A candidate can lose, and that is a useful result. Do not add isolated timings into a serving latency breakdown.'''),
code('benchmark = study.benchmark()'),
md('''## 5. Table preparation and refresh costs
Measure creating the 257-entry lookup table and refreshing an existing table using build-plus-copy. Test eager and graph execution over three trials. These add **12 timing rows and 132 samples**, bringing the total to **588 rows / 6,468 samples**.

This exposes the caching tradeoff: a fast cached lookup is useful only if preparation can be reused. Refresh is measured separately because a cached graph reads a fixed pointer. There is no automatic break-even claim; review setup cost, savings, mapping lifetime and repeatability together.'''),
code('setup = study.setup_cost()'),
md('''## 6. Separate trace
After timing, trace six iterations of each variant at 128 tokens, int64 input, hot-local routes, including the fallback cast. Labels are enabled only here. Variant order is shuffled in each trace iteration.

`trace_device_operations.csv` contains actual kernels/copies/memsets, excluding annotation ranges. `trace_variants.csv` links them to the variant using launch correlation IDs and preserves each iteration. A CPU-only trace is marked partial, not reported as zero GPU cost. Trace durations are not directly comparable to the unprofiled benchmark.'''),
code('profiler = study.profile()'),
md('''## 7. Export and return the ZIP
The archive contains configuration and versions, correctness status, raw samples, timing summaries, per-trial comparisons, setup/refresh timings, trace evidence, source provenance and checksums. Ratios in `comparisons.csv` are baseline divided by candidate: above 1 favors the candidate, below 1 favors the baseline.

We will review correctness first, then repeated graph results and eager results, then preparation costs. A consistent isolated win would justify an integration review; it would not establish full serving or H200 speedup. No candidate is automatically promoted.

Download via the link below or Kaggle's output/file panel in `/kaggle/working`. Keep this ZIP separate from the previous notebook outputs.'''),
code('archive = study.export()\nfrom IPython.display import FileLink, display\ntry:\n    link = archive.relative_to(Path.cwd())\nexcept ValueError:\n    link = archive\ndisplay(FileLink(str(link)))'),
md('''Method references: [PyTorch CUDA graph semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs) and [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html). The experiment records installed versions and uses only PyTorch operations; no Triton compilation is required.'''),
], metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name':'python','version':'3.10'}})
nb.cells[2].metadata['jupyter'] = {'source_hidden': True}
for cell in nb.cells:
    if cell.cell_type=='code': ast.parse(cell.source)
nbformat.validate(nb)
nbformat.write(nb, HERE/'03_kaggle_mapping_candidate.ipynb')
(HERE/'source_manifest_03.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(HERE/'mapping_helper.py').write_text(helper,encoding='utf-8')
print('Built', len(nb.cells), 'cells;', sum(c.cell_type=='code' for c in nb.cells), 'code cells')
