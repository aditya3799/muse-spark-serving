import ast,hashlib,json
from pathlib import Path
import nbformat
HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\OpenKernelChallenge\serving');SOURCE=ROOT/'profiling_kaggle/notebook10_support'
files=['timing_support.py','source_slices.py','input_harness.py','ownership_study.py','update_study.py','contract_study.py','mapping_helper.py']
for n in files:(HERE/n).write_text((SOURCE/n).read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
tree=ast.parse((HERE/'source_slices.py').read_text())
classes={n.name:n for n in tree.body if isinstance(n,ast.ClassDef)}
forward=lambda name:ast.dump(next(n for n in classes[name].body if isinstance(n,ast.FunctionDef) and n.name=='forward'),include_attributes=False)
assert forward('CandidateSourceSlice')==forward('SharedSourceSlice')
rules=(ROOT/'RULES.md').read_text(encoding='utf-8');assert '--ep-size 8' in rules
audit={'candidate_forward_ast_identical':True,'rules_sha256':hashlib.sha256(rules.encode()).hexdigest(),
    'judge_launch_ep_evidence':[{'line':i,'source':s.strip()} for i,s in enumerate(rules.splitlines(),1) if '--ep-size' in s],
    'scope':'Judge launch specification is source evidence, not a runtime observation. Same forward arithmetic; ownership/refresh contracts differ.'}
(HERE/'CACHE_CHOICE_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
manifest={'parent_source_manifest':json.loads((SOURCE/'source_manifest_10.json').read_text()),'cache_choice_audit':audit,
    'scope':'Unchanged existing drafts, static map ownership comparison using int32 IDs and float32 scores.',
    'file_sha256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in files+['cache_study.py']}}
(HERE/'source_manifest_11.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
nb=nbformat.v4.new_notebook(cells=[
md('''# 11 - Choose a cache design using native routing types

Notebook 10 confirmed mapping gains with int32 expert IDs and float32 scores. This experiment compares the two existing cache designs under those input types, including eager execution where their host behavior can differ.

**Kaggle: upload this file, select GPU T4, and Run All.** Only GPU 0 is used. No packages, Internet, weights, datasets or credentials are needed. Keep the notebook private if you do not want to publish its embedded project source slices.

Download `v4_cache_choice_*.zip` into `profiling_kaggle/notebook11_results`. Run export even if an earlier gate fails, then stop the GPU session.

Active V4 and both drafts remain unchanged. This is runtime MoE preparation, not complete model or startup profiling.'''),
md('''## 1. Performance and integration tradeoffs

| Implementation | Map representation | Update contract |
|---|---|---|
| Baseline | Original registered map, offset zero | Direct map writes |
| Two-buffer | Original registered map plus separate lookup | Refresh lookup after any map edit |
| Single-owner | Lookup buffer; map is a property view at offset one | Map writes immediately affect lookup |

The candidate forward methods have identical syntax trees. Their mapping arithmetic is unchanged; storage ownership differs. The two-buffer design preserves the original map registration and pointer offset, while the single-owner design avoids stale duplicated contents. Neither property establishes full framework or actual fused-kernel compatibility.

The local judge launch specification contains `--ep-size 8`; the tested map holds 32 of 256 experts, with IDs 64..95 local. Each stage has a distinct module with that fixed contiguous map. This is still synthetic source-prefix execution on one GPU, not actual distributed expert execution.'''),
*[code((HERE/n).read_text(encoding='utf-8')) for n in files[:6]],
code((HERE/'cache_study.py').read_text(encoding='utf-8')),
code('SOURCE_MANIFEST='+repr(manifest)+'\nMAPPING_HELPER_SOURCE='+repr((HERE/'mapping_helper.py').read_text(encoding='utf-8'))+'\n\nSAMPLES=11\nREPEATS=3\nTRIALS=3\nstudy=CacheStudy(SOURCE_MANIFEST,MAPPING_HELPER_SOURCE,samples=SAMPLES,repeats=REPEATS,trials=TRIALS)\nprint("Results directory:",study.out)'),
md('''## 2. Correctness and ownership gates

Re-run inherited lifecycle and input-contract tests: 576 + 80 output checks, including empty inputs, layouts, dtypes, device moves and checkpoint exclusion.

The new preservation gate verifies named buffers and storage offsets. It captures the two candidate prefixes, directly changes their maps, demonstrates stale two-buffer output, then verifies explicit refresh and checked updates restore correct output with stable map storage. It also verifies direct writes are immediately visible to the single-owner graph. CPU-only execution performs the equivalent eager checks and skips GPU timing.

Small fused branches are only checked for selection/metadata. No uninitialized ID values are read and the real fused kernel is not executed.'''),
code('environment=study.environment()\nlifecycle=study.lifecycle()\ninput_contract=study.input_contract()\npreservation_contract=study.preservation_contract()'),
md('''## 3. Static-map benchmark

Token counts **9/32/128/512**, **1/8 independent stages**, and **hot-local/uniform** routing distributions. Hot-local contains four routes from the local range per token; uniform samples from all 256 experts. Scores are 0.125, already float32; IDs are contiguous int32. Hidden activations remain float16 width-32 placeholders.

Three shuffled trials produce **48 cases**. Three implementations times eager/combined-graph execution give six variants per case. Eleven samples and three complete chains per sample give **288 timing rows and 3,168 raw samples**. All six variants are interleaved within every sample.

Before timing, changed sentinel and all-remote routes exercise every stage: **2,592 output checks**. The requested timing distribution is then restored; outputs are checked again after timing. Maps remain fixed throughout timing. Setup, refresh tests, capture and validation are excluded.

Measure CUDA-event, synchronized wall and host enqueue time per chain. These overlap and cannot be added/subtracted for exact CPU/GPU attribution. Compare paired absolute microsecond savings and ratios, especially between the two candidates; small differences do not establish equivalence. Stages are independent mapping prefixes with no intervening model computation.'''),
code('benchmark=study.benchmark()'),
md('''## 4. Separate static trace

Profile all six variants at 128 tokens and eight stages, three cycles each. Check whether both candidates launch the same mapping work and whether they differ from baseline. No refresh occurs in this trace. Only actual kernels/copies count; missing GPU support or graph correlation is reported as partial.'''),
code('profiler=study.profile()'),
md('''## 5. Export and decision criteria

The ZIP contains timings, raw samples, **288 comparisons**, lifecycle/update evidence, trace, provenance and checksums. Comparisons include both candidates versus baseline and single-owner versus two-buffer, for each execution schedule.

If static performance is comparable, the original registered-map contract may favor the two-buffer design, subject to an explicit refresh lifecycle. If updates are frequent, earlier update-overhead results also matter; this static experiment does not remeasure update cost or prove real update frequency.

Neither draft is promoted. Actual model loading, buffer integrations, small fused alignment, expert GEMMs, communication, scheduler/tokenizer initialization and H200/full-serving performance remain unvalidated.'''),
code('archive=study.export()\nfrom IPython.display import FileLink,display\ntry:\n    link=archive.relative_to(Path.cwd())\nexcept ValueError:\n    link=archive\ndisplay(FileLink(str(link)))'),
],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.10'}})
for c in nb.cells:
    if c.cell_type=='code':ast.parse(c.source)
for c in nb.cells[2:9]:c.metadata['jupyter']={'source_hidden':True}
nbformat.validate(nb);nbformat.write(nb,HERE/'11_kaggle_cache_choice.ipynb')
print('Built',len(nb.cells),'cells')
