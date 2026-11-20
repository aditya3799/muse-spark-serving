import ast,hashlib,json
from pathlib import Path
import nbformat

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\OpenKernelChallenge\serving\profiling_kaggle')
prior=ROOT/'notebook05_support'
core=(prior/'timing_support.py').read_text(encoding='utf-8')
slices=(prior/'source_slices.py').read_text(encoding='utf-8')
helper=(prior/'mapping_helper.py').read_text(encoding='utf-8')
source=(prior/'source_study.py').read_text(encoding='utf-8');lines=source.splitlines()
klass=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='SourceStudy')
methods=['\n'.join(lines[n.lineno-1:n.end_lineno]) for n in klass.body if isinstance(n,ast.FunctionDef) and n.name in ('pair','inputs','set_inputs','check_result')]
harness='''"""Input/check helpers extracted unchanged from validated Notebook 05."""
import torch
if 'ExperimentCore' not in globals():
    from timing_support import ExperimentCore,routes,reference,expert_mapping
if 'BaselineSourceSlice' not in globals():
    from source_slices import BaselineSourceSlice,CandidateSourceSlice

class SourceInputs(ExperimentCore):
'''+ '\n\n'.join(methods)+'\n'
support=(HERE/'replay_study.py').read_text(encoding='utf-8')
for name,text in [('timing_support.py',core),('source_slices.py',slices),('mapping_helper.py',helper),('input_harness.py',harness)]:
    (HERE/name).write_text(text,encoding='utf-8',newline='\n')
manifest=json.loads((prior/'source_manifest_05.json').read_text())
assert hashlib.sha256(slices.encode()).hexdigest()==manifest['source_slices_sha256']
manifest['notebook05_study_sha256']=manifest.pop('study_sha256')
manifest.update(input_harness_sha256=hashlib.sha256(harness.encode()).hexdigest(),study_sha256=hashlib.sha256(support.encode()).hexdigest(),
    experiment='Inference-mode replay schedule; unchanged candidate/source slices from Notebook 05')
(HERE/'source_manifest_06.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
nb=nbformat.v4.new_notebook(cells=[
md('''# 06 - Refresh outside an existing CUDA graph

Notebook 05 found a consistent cached-prefix win, but updating every eager call regressed. Its graph reuse test captured the update **inside** the graph. Here we test the missing schedule: refresh the map from Python, then replay an already-captured prefix graph.

**Kaggle:** upload this file, select **GPU T4**, keep defaults and **Run All**. Only GPU 0 is used. Internet can stay off; no installs, weights, datasets or credentials are needed. All code is embedded. Keep the notebook private if you do not want to publish the embedded source slices.

This uses the **unchanged, unapplied Notebook 05 draft**. It runs only source-selected pre-alignment code, not a full serving graph, aligner or GEMM. No serving patch is applied.

Download `v4_replay_schedule_*.zip` and place it in `profiling_kaggle/notebook06_results`. Run the export cell even if a section fails or the profiler is partial; stop the GPU session after downloading.'''),
md('''## 1. Three schedules, one candidate

| Scenario | Work in one measured cycle |
|---|---|
| `static` | No map update; launch the existing prefix graph K times |
| `external` | Eager map update (including candidate refresh), then K separate prefix-graph launches |
| `captured` | Replay one larger graph containing the update and K prefixes |

**Primary comparison:** baseline versus candidate within `external`. `static` supplies the reused-map control. `captured` shows the graph structure used in Notebook 05, measured within this same run.

All calls run under `torch.inference_mode()`. Benchmark module buffers are created normally before inference begins. Correctness also tests modules created inside inference mode, with their updates performed inside that context.

Do not treat differences across the three scenarios as pure cache effects: they also change the number of host graph launches. These are prefix graphs, not full-model graphs.'''),
code(core),code(slices),code(harness),code(support),
code('SOURCE_MANIFEST='+repr(manifest)+'\nMAPPING_HELPER_SOURCE='+repr(helper)+'\n\nSAMPLES=11\nREPEATS=3\nTRIALS=3\nstudy=ReplayStudy(SOURCE_MANIFEST,MAPPING_HELPER_SOURCE,samples=SAMPLES,repeats=REPEATS,trials=TRIALS)\nprint("Results directory:",study.out)'),
md('''## 2. Environment
Record software/GPU versions and a clock snapshot. Compare variants within matching trials. No upgrades or driver changes are attempted. CPU-only mode runs correctness and skips GPU measurements.'''),
code('environment=study.environment()'),
md('''## 3. Inference-mode correctness
Check **192 input configurations / 384 outputs**, covering normal and inference-mode buffer creation, int32/int64 IDs, 9/32/128/256 tokens, varied expert mappings, dropped routes and other negative IDs. All these shapes use the actual standalone mapping fallback in the source slice.

These tests do not authorize mutation of inference-created tensors outside inference mode. Direct map edits still require an explicit lookup refresh, and buffer replacement or relocation requires graph recapture.'''),
code('correctness=study.correctness()'),
md('''## 4. Replay schedule experiment
Use 32/128 tokens with K = **1, 2, 4, 8 or 64** graph replays per cycle, int64 IDs and hot-local routing. Three shuffled trials and six interleaved variants give **30 cases, 180 timing rows, 1,980 raw samples**, with **360 changed-input schedule checks** before timing and checks afterward.

Each of 11 samples averages three complete cycles. Timing rows contain **microseconds per cycle**; comparisons also divide by K for cost per prefix including the update share.

Incoming map values are already on GPU. Input generation/copies, graph capture and correctness checks are excluded. `external` includes the eager update and lookup refresh, while `static` includes neither. During timing all variants use matching mapping values, so randomized variant order cannot invalidate the static control. The stated update still runs every cycle even if values repeat.

No profiler or annotation labels are active in timing calls. CUDA-event intervals may include launch gaps; synchronized wall time also captures host overhead. We are measuring controlled reuse, not claiming a production update frequency or universal crossover point.'''),
code('benchmark=study.benchmark()'),
md('''## 5. Separate external-update trace
Trace four cycles per variant at 128 tokens and K=4, with the map update outside capture. Actual graph kernels may lack host-launch correlation on some profiler versions; unmatched events remain in the exported CSV and result in `partial` status. No stage totals are invented. Benchmark timing does not depend on profiler support.'''),
code('profiler=study.profile()'),
md('''## 6. Export
Return the entire ZIP. It contains correctness, raw timing samples, per-trial comparisons for each schedule, trace evidence, versions and checksums.

The next review will evaluate the external-update results before proposing any change in the cache policy, and include the next experiment. The same V4-relative patch remains unapplied.

Still unmeasured: full model initialization/loading, actual alignment/GEMMs, multi-GPU communication, scheduler/tokenizer startup and H200 latency.'''),
code('archive=study.export()\nfrom IPython.display import FileLink,display\ntry:\n    link=archive.relative_to(Path.cwd())\nexcept ValueError:\n    link=archive\ndisplay(FileLink(str(link)))'),
md('''Method references: [PyTorch inference mode](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html), [inference-tensor update restrictions](https://github.com/pytorch/pytorch/blob/main/c10/core/TensorImpl.h), and [CUDA graph semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs).'''),
],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.10'}})
for i in (2,3,4,5):nb.cells[i].metadata['jupyter']={'source_hidden':True}
for c in nb.cells:
    if c.cell_type=='code':ast.parse(c.source)
nbformat.validate(nb);nbformat.write(nb,HERE/'06_kaggle_replay_schedule.ipynb')
print('Built',len(nb.cells),'cells;',sum(c.cell_type=='code' for c in nb.cells),'code cells')
