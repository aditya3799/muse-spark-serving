import ast,hashlib,json
from pathlib import Path
import nbformat
HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\OpenKernelChallenge\serving\profiling_kaggle')
prior=ROOT/'notebook06_support'
core=(prior/'timing_support.py').read_text(encoding='utf-8')
harness=(prior/'input_harness.py').read_text(encoding='utf-8')
helper=(prior/'mapping_helper.py').read_text(encoding='utf-8')
slices=(HERE/'source_slices.py').read_text(encoding='utf-8')
support=(HERE/'ownership_study.py').read_text(encoding='utf-8')
for name,text in [('timing_support.py',core),('input_harness.py',harness),('mapping_helper.py',helper)]:
    (HERE/name).write_text(text,encoding='utf-8',newline='\n')
manifest=json.loads((HERE/'source_manifest_07.json').read_text())
manifest.update(timing_support_sha256=hashlib.sha256(core.encode()).hexdigest(),input_harness_sha256=hashlib.sha256(harness.encode()).hexdigest(),study_sha256=hashlib.sha256(support.encode()).hexdigest())
(HERE/'source_manifest_07.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
nb=nbformat.v4.new_notebook(cells=[
md('''# 07 - One owning buffer instead of a refresh copy

Notebook 06 showed that cached mapping helps static maps, but eager refresh outside an existing graph can be expensive. This experiment tests a structural change: make the expert map a view into the lookup's storage, so updating the map updates the lookup directly.

**Kaggle:** upload this file, select **GPU T4**, keep defaults and **Run All**. Only GPU 0 is used. Internet can stay off; no installs, model weights, datasets or credentials are needed. The notebook is self-contained. Keep it private if you do not want to publish project source slices.

Both drafts are **unapplied**. This executes source-selected initialization and pre-alignment prefixes, not the complete serving class, real alignment/GEMMs or a full-model graph.

Download `v4_single_owner_*.zip` and place it in `profiling_kaggle/notebook07_results`. Run the export cell even if a section fails, then stop the GPU session.'''),
md('''## 1. Three implementations

| Variant | Storage and updates |
|---|---|
| `baseline` | Original map; update copies it once; original mapping helper |
| `two_buffer` | Prior map plus lookup; update copies map, fills sentinel, copies map into lookup |
| `single_owner` | Lookup alone; `_expert_map` is a view of `lookup[1:]`; update copies into that view once |

The two candidate variants retain **identical forward lookup arithmetic**. This isolates ownership/update behavior, including the host cost of creating a view on property access.

Compatibility changes are explicit: `_expert_map` is no longer a named buffer, attribute replacement is rejected, and the map view starts one int32 element (four bytes) into the allocation. Device moves should remain safe because the property derives a fresh view of the moved owner. Graphs still require recapture after relocation. Actual fused-kernel behavior with the changed pointer offset is **not tested here**.

The new V4-relative draft is saved locally in `notebook07_support`; active V4 and the prior draft remain unchanged.'''),
code(core),code(slices),code(harness),code(support),
code('SOURCE_MANIFEST='+repr(manifest)+'\nMAPPING_HELPER_SOURCE='+repr(helper)+'\n\nSAMPLES=11\nREPEATS=3\nTRIALS=3\nstudy=OwnershipStudy(SOURCE_MANIFEST,MAPPING_HELPER_SOURCE,samples=SAMPLES,repeats=REPEATS,trials=TRIALS)\nprint("Results directory:",study.out)'),
md('''## 2. Environment
Record versions and GPU details without changing installed packages. CPU-only execution runs lifecycle checks and skips CUDA timings. Compare variants within this run.'''),
code('environment=study.environment()'),
md('''## 3. Correctness and ownership gate

- 192 input configurations / **576 outputs** across three implementations, both normal/inference buffer creation, int32/int64, varied mappings and negative IDs.
- Prove map and lookup share storage; verify the four-byte offset and contiguous view.
- Write directly through the map view and verify correct output **without refresh**; reject replacing the map attribute.
- Move CPU → GPU → CPU → GPU before capture and verify aliasing/dtype.
- Reject malformed updates before writes.
- Check nonpersistent checkpoint exclusion using a marker, and measure registered storage: baseline **1,024 bytes**, two-buffer **2,052 bytes**, single-owner **1,028 bytes** for this mapping size.
- Check small-input branch selection and no-map behavior. Deferred fused-branch local-ID values are not inspected.
- Capture a graph, directly update the map view, and verify replay sees new values with stable owning storage.

All updates/forwards use inference mode. Named-buffer compatibility, full checkpoints/meta loading and real fused kernels remain outside this gate.'''),
code('lifecycle=study.lifecycle()'),
md('''## 4. Static and external-update schedules
At 32/128 tokens, int64 IDs and hot-local routes, compare K = **1, 2, 4, 8 or 64** separate prefix-graph replays per cycle.

`static` performs no update. `external` updates from Python before those graph launches; candidate update cost is fully included. No whole-cycle captured-update control is included this time.

Three shuffled trials × ten configurations × six variants produce **30 cases, 180 timing rows and 1,980 raw samples**, with **360 changed-input schedule checks** before timing and checks afterward. Each sample averages three complete cycles; 11 samples are taken per variant. Variant order is interleaved within every sample.

Raw timing units are microseconds per cycle. Comparison rows also divide by K to include update share per replay. Incoming map values already reside on GPU; input generation, capture and correctness checks are outside timing. No labels or profiler are active during timing.

The new candidate may still lose: the update API retains validation/decorator overhead, and its property creates a view. Inspect matched trials rather than assuming one copy guarantees a win.'''),
code('benchmark=study.benchmark()'),
md('''## 5. Separate external-update trace
Profile four external-update cycles per implementation at 128 tokens and K=4. Check whether the single-owner variant removes the extra fill/copy seen previously. GPU annotation spans are excluded; missing graph correlation produces partial status with unmatched events retained. Timing is independent of profiler support.'''),
code('profiler=study.profile()'),
md('''## 6. Export
Return the entire ZIP for the review and next experiment. It contains lifecycle evidence, timings, comparisons, raw samples, trace data, provenance and checksums.

This does not promote either draft or establish full-serving/H200 performance. It tests whether one owning buffer solves the refresh overhead and stale-copy problem under the measured schedule. Actual alignment/GEMMs, framework-wide buffer handling, scheduler/tokenizer startup and model loading remain unmeasured.'''),
code('archive=study.export()\nfrom IPython.display import FileLink,display\ntry:\n    link=archive.relative_to(Path.cwd())\nexcept ValueError:\n    link=archive\ndisplay(FileLink(str(link)))'),
],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.10'}})
for i in (2,3,4,5):nb.cells[i].metadata['jupyter']={'source_hidden':True}
for c in nb.cells:
    if c.cell_type=='code':ast.parse(c.source)
nbformat.validate(nb);nbformat.write(nb,HERE/'07_kaggle_single_owner.ipynb')
print('Built',len(nb.cells),'cells;',sum(c.cell_type=='code' for c in nb.cells),'code cells')
