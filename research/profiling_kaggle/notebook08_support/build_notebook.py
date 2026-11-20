import ast,hashlib,json
from pathlib import Path
import nbformat
HERE=Path(__file__).resolve().parent
SOURCE=Path(r'D:\OpenKernelChallenge\serving\profiling_kaggle\notebook07_support')
files=['timing_support.py','source_slices.py','input_harness.py','ownership_study.py','mapping_helper.py']
for name in files:(HERE/name).write_text((SOURCE/name).read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
manifest={'parent_source_manifest':json.loads((SOURCE/'source_manifest_07.json').read_text()),
    'scope':'Unchanged source slices and single-owner draft from Notebook 07. Update API controls are experiment-only; no new serving patch.',
    'file_sha256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in files+['update_study.py']}}
(HERE/'source_manifest_08.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
nb=nbformat.v4.new_notebook(cells=[
md('''# 08 - Where does the remaining update overhead come from?

Notebook 07 removed the duplicated refresh copy, but one update per replay still lost against the baseline. This experiment holds the mapping arithmetic fixed and changes only the way Python submits map updates.

**Upload this file to Kaggle, select GPU T4, and Run All.** Only GPU 0 is used. No installs, Internet, model weights, datasets or credentials are needed. Keep the notebook private if you do not want to publish its embedded project source slices.

Download `v4_update_overhead_*.zip` and place it in `profiling_kaggle/notebook08_results`. Run the export cell even if an earlier check fails. Stop the GPU session after downloading.

This is a diagnostic experiment. Active V4 and both candidate drafts remain unchanged. There is no full-model, real alignment/GEMM or H200 execution.'''),
md('''## 1. Matched controls

| Variant | Update before K separate graph replays |
|---|---|
| `baseline_static` | None; original mapping graph |
| `single_static` | None; single-owner mapping graph |
| `baseline_direct` | Original map attribute, one direct copy |
| `baseline_checked` | Identical decorated update method used by the single-owner class |
| `single_checked` | Existing single-owner checked method, including validation and property access |
| `single_property` | One direct copy through the map property; creates the view on every call |
| `single_bound` | One direct copy through a view retained after device placement |

Baseline checked is a harness-only subclass assigning the **same method object** used by the candidate. The baseline forward and buffers are unchanged. This makes the cost of the checked API visible on both implementations.

Direct-copy paths require validated shape, dtype, device and stable storage. We validate those conditions before timing, not on every copy. Retaining a view is a diagnostic lower-overhead control, not a safe integration by itself: moving or replacing the owner requires rebinding the view and recapturing graphs.'''),
*[code((HERE/n).read_text(encoding='utf-8')) for n in files[:4]],
code((HERE/'update_study.py').read_text(encoding='utf-8')),
code('SOURCE_MANIFEST='+repr(manifest)+'\nMAPPING_HELPER_SOURCE='+repr((HERE/'mapping_helper.py').read_text(encoding='utf-8'))+'\n\nSAMPLES=11\nREPEATS=3\nTRIALS=3\nstudy=UpdateStudy(SOURCE_MANIFEST,MAPPING_HELPER_SOURCE,samples=SAMPLES,repeats=REPEATS,trials=TRIALS)\nprint("Results directory:",study.out)'),
md('''## 2. Environment and correctness gates

The inherited Notebook 07 lifecycle checks still cover all three implementations: 192 configurations and 576 output checks, storage aliasing, moves, graph visibility, small-branch selection and checkpoint exclusion. The actual small fused kernel is omitted, so its output and offset-pointer behavior are not validated.

An additional gate checks the identical checked-method control, rejects invalid shapes/dtypes, and demonstrates that an old retained view becomes stale after owner replacement. CPU-only execution runs these gates and skips GPU timing.'''),
code('environment=study.environment()\nlifecycle=study.lifecycle()\nupdate_contract=study.update_contract()'),
md('''## 3. Timing the update paths

32 and 128 tokens, int64 IDs, K = **1, 2, 4, 8** replays per cycle. Three shuffled trials, seven variants, eleven samples and three complete cycles per sample: **24 cases, 168 timing rows, 1,848 raw samples**.

Before timing, each variant is checked against three changed maps. The current map is deliberately overwritten with -1, and the input includes a local expert whose correct result differs from that poison. Update variants must repair it themselves; static controls receive their expected map before replay. This gives **504 state checks**, including 360 checks of actual update variants. Each variant is checked again after timing.

Every sample records:

- **CUDA-event time:** elapsed device timeline for update plus K replays; it can include idle time waiting for Python to submit work.
- **Synchronized wall time:** host time through completion, including event record/synchronization overhead.
- **Host enqueue time:** time spent calling the update and replay APIs, excluding the event calls and final synchronization.

These measurements overlap. Do not add enqueue and event times, or subtract them to infer exact CPU/GPU percentages. Units are microseconds per complete cycle. Compare variants within the same trial/configuration; CPU preparation, graph capture, contract checks and correctness checks are outside timing. Incoming maps already reside on the GPU. Profiling is disabled during timing.'''),
code('benchmark=study.benchmark()'),
md('''## 4. Separate device trace

At 128 tokens and K=1, trace four cycles for each of the five update paths. Check whether each has one copy and whether same-implementation paths launch the same prefix kernels. This cannot measure Python validation cost by counting kernels. Missing CUDA trace support or graph correlation is reported as partial.'''),
code('profiler=study.profile()'),
md('''## 5. Export

The ZIP includes raw samples, all three timing metrics, 192 matched comparison rows, lifecycle/contract evidence, trace, provenance and checksums.

Interpretation: if checked versus direct paths differ while their device operations match, host submission overhead is implicated. If retained-view and property-copy differ, view creation is implicated. These are controlled comparisons, not an additive latency decomposition. A faster bypass does not justify removing production validation.

Full serving integration, actual fused alignment/GEMMs, framework-wide buffer introspection, scheduler/tokenizer initialization and H200 performance remain unmeasured.'''),
code('archive=study.export()\nfrom IPython.display import FileLink,display\ntry:\n    link=archive.relative_to(Path.cwd())\nexcept ValueError:\n    link=archive\ndisplay(FileLink(str(link)))'),
],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.10'}})
for c in nb.cells:
    if c.cell_type=='code':ast.parse(c.source)
for c in nb.cells[2:7]:c.metadata['jupyter']={'source_hidden':True}
nbformat.validate(nb);nbformat.write(nb,HERE/'08_kaggle_update_overhead.ipynb')
print('Built',len(nb.cells),'cells')
