import ast,hashlib,json
from pathlib import Path
import nbformat
HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\OpenKernelChallenge\serving\profiling_kaggle')
source=ROOT/'11_kaggle_cache_choice.ipynb';validation=json.loads((ROOT/'LOCAL_VALIDATION_11.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==validation['notebook_sha256']
assert all(validation['statuses'][s]=='passed' for s in ('environment','lifecycle','input_contract','preservation_contract','benchmark'))
nb=nbformat.read(source,as_version=4);original_code=[c.source for c in nb.cells if c.cell_type=='code']
nb.cells[0].source='''# 14 - Replicate the cache-choice benchmark on Kaggle

The earlier performance experiments ran on **Colab T4**. After Colab limits were exhausted, the workflow moved to **Kaggle T4** with a different installed software stack. Notebooks 12 and 13 passed correctness gates there; they did not remeasure latency.

This notebook reruns the **exact executable cells and default configuration of Notebook 11** on the current environment. No algorithm, measurement logic or source manifest has changed. This isolates whether the earlier cache-choice conclusion holds on Kaggle.

**Upload this file to Kaggle, select GPU T4, keep defaults, and Run All.** Only GPU 0 is used even if two GPUs are visible. No packages, Internet, weights, datasets or credentials are needed. Keep it private if you do not want to publish its embedded project source slices.

Download the resulting **`v4_cache_choice_*.zip`** into **`profiling_kaggle/notebook14_results`**. The original ZIP prefix is intentionally retained because the experiment code is unchanged. Export even after an earlier failure, then stop the GPU session.

Active V4 and both drafts remain unapplied/unchanged. The workload is isolated runtime MoE preparation, not full-model or startup execution.'''
nb.cells[-2].source='''## 5. Export and replication decision

Return the complete **`v4_cache_choice_*.zip`** in **`profiling_kaggle/notebook14_results`**. Expected: **48 cases, 288 timing rows, 3,168 raw samples, 2,592 changed-input output checks and 288 comparisons**.

The key comparisons are two-buffer versus baseline in eager and graph schedules, plus single-owner versus two-buffer. Evaluate each against its matched baseline in this Kaggle run. Then compare the direction and distribution of matched ratios with the earlier Colab run; do not treat changes in absolute microseconds across providers as code improvements.

The runtime records installed versions and GPU details without changing them. Keep defaults to preserve comparability. The original experiment's export schema and summary title are retained because executable cells are identical to Notebook 11.

The two-buffer draft is still conditional on correct refresh after map edits. A successful replication does not validate actual fused kernels, full serving integration, H200 performance or scheduler/tokenizer initialization. Neither draft is applied or promoted.'''
code=[c for c in nb.cells if c.cell_type=='code']
assert [c.source for c in code]==original_code
assert all(c.execution_count is None and not c.outputs for c in code)
for c in code:ast.parse(c.source)
nbformat.validate(nb);target=HERE/'14_kaggle_cache_replication.ipynb';nbformat.write(nb,target)
roundtrip=nbformat.read(target,as_version=4);assert [c.source for c in roundtrip.cells if c.cell_type=='code']==original_code
record={'schema':'passed','validation_method':'Exact executable-cell equivalence to the previously GPU-validated Notebook 11; only Markdown changed. No new local GPU run.',
    'source_notebook':str(source),'source_notebook_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'source_validation_sha256':hashlib.sha256((ROOT/'LOCAL_VALIDATION_11.json').read_bytes()).hexdigest(),
    'code_cell_sha256':[hashlib.sha256(s.encode()).hexdigest() for s in original_code],
    'code_cells':len(original_code),'code_and_configuration_unchanged':True,'clean_outputs':True,
    'source_gpu_validation_statuses':validation['statuses'],'source_gpu_validation_totals':validation['totals'],
    'notebook_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'result_prefix':'v4_cache_choice_','results_folder':'notebook14_results',
    'intent':'Current Kaggle within-run performance replication of earlier Colab cache-choice benchmark. No new algorithm or active source change.'}
(HERE/'LOCAL_VALIDATION_14.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
