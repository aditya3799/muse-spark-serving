import ast,hashlib,json
from pathlib import Path
import nbformat
HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\OpenKernelChallenge\serving');SOURCE=ROOT/'profiling_kaggle/notebook11_support';DRAFT=ROOT/'profiling_kaggle/notebook05_support'
files=['timing_support.py','source_slices.py','input_harness.py','mapping_helper.py']
for n in files:(HERE/n).write_text((SOURCE/n).read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
selected=json.loads((DRAFT/'source_manifest_05.json').read_text())
for src,dst,key in [('cached_mapping_v4_draft.diff','selected_two_buffer_v4.diff','draft_diff_sha256'),('candidate_muse_spark_fp4_moe.py','selected_two_buffer_source.py','candidate_source_sha256')]:
    content=(DRAFT/src).read_text(encoding='utf-8');assert hashlib.sha256(content.encode()).hexdigest()==selected[key]
    (HERE/dst).write_text(content,encoding='utf-8',newline='\n')
def classes(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef)}
old,new=classes(DRAFT/'source_slices.py'),classes(HERE/'source_slices.py')
for name in ('BaselineSourceSlice','CandidateSourceSlice'):assert old[name]==new[name]
manifest={'selection':'Existing two-buffer draft selected for further static-map correctness validation; unapplied.',
    'baseline_patch_sha256':selected['baseline_patch_sha256'],'baseline_owner_sha256':selected['baseline_owner_sha256'],
    'selected_source_sha256':selected['candidate_source_sha256'],'selected_diff_sha256':selected['draft_diff_sha256'],
    'source_slices_match_original_selected_classes':True,'original_draft_manifest':selected,
    'file_sha256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in files+['gate_study.py']}}
(HERE/'source_manifest_12.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
nb=nbformat.v4.new_notebook(cells=[
md('''# 12 - Correctness gate for the selected two-buffer draft

Notebook 11 favors the earlier two-buffer cache for static maps: it preserves the registered expert map and avoids the single-owner property's eager overhead. This notebook freezes that **unchanged, unapplied draft** and expands correctness coverage. It does not measure latency.

**Kaggle: upload this file, select GPU T4, and Run All.** Only GPU 0 is used. No installation, Internet, datasets, weights or credentials are required. Keep the notebook private if you do not want to publish embedded project source slices.

Download `v4_two_buffer_gate_*.zip` into `profiling_kaggle/notebook12_results`. Run export even if a gate fails, then stop the GPU session.

The selected V4-relative diff and full candidate source are saved locally under `notebook12_support`; the notebook embeds only selected constructor/pre-alignment code. **Active V4 remains unchanged.** Actual fused kernels, GEMMs and full serving are not executed.'''),
md('''## 1. What is selected, and what still needs proof?

The two-buffer draft retains the original `_expert_map` buffer at offset zero and maintains a separate lookup with a leading -1 sentinel. Its cached lookup saves mapping operations for the standalone fallback path.

**Required lifecycle:** any map edit must refresh the lookup before consumers; the checked update method performs both copies. Maps must be updated before graph replay on an ordered stream. Buffer replacement or movement requires recapture. Do not refresh on every forward by default; frequent refresh has previously regressed performance.

Both selected source-slice classes are verified to match the original Notebook 05 draft by syntax tree. The diff and full source hashes are frozen. Selecting the draft for this test does not apply it or establish full serving compatibility.'''),
*[code((HERE/n).read_text(encoding='utf-8')) for n in files[:3]],
code((HERE/'gate_study.py').read_text(encoding='utf-8')),
code('SOURCE_MANIFEST='+repr(manifest)+'\nMAPPING_HELPER_SOURCE='+repr((HERE/'mapping_helper.py').read_text(encoding='utf-8'))+'\n\nstudy=GateStudy(SOURCE_MANIFEST,MAPPING_HELPER_SOURCE)\nprint("Results directory:",study.out)'),
md('''## 2. Environment and all-partition matrix

Eight partitions of 32 local experts out of 256, plus a no-map case. Test **0/1/2/8/9/16/64/128/512 tokens**, int32/int64 IDs, contiguous/strided ID layouts, and local/remote/mixed/all-negative routing. Mixed routes include global boundary IDs and -1, -2, int32-min sentinels. Scores are float32; hidden activations are float16 width-32 placeholders.

Expected **1,296 input configurations / 2,592 baseline-and-candidate output checks**. On CUDA, 384 outputs take the existing small fused branch and receive metadata-only checks. The other **2,208** have their ID values checked against an independent Python reference. No uninitialized fused ID values are read. CPU uses the fallback for all outputs.

This is eight partition mappings executed sequentially on one GPU, not an eight-GPU run. Positive expert IDs stay within 0..255; invalid positive IDs are outside the router contract.'''),
code('environment=study.environment()\nmatrix=study.matrix()'),
md('''## 3. Changed maps through captured graphs

For every partition, capture baseline and candidate at 32 tokens. Exercise initial, all-remote, reversed-local and next-partition maps through both checked updates and explicit map-copy-plus-refresh.

Before each update, overwrite old maps and cached lookup contents with -7 so a missed update cannot pass. Expect **64 update cycles, 128 graph output checks and 64 stable-pointer checks**. Updates occur before replay on the same ordered stream; concurrent/cross-stream updates remain outside scope.'''),
code('updates=study.update_gate()'),
md('''## 4. Device moves and recapture

For all eight partitions, validate an initial graph, release old graphs, move both sliced modules from GPU to CPU, validate CPU output, move back to GPU, and capture fresh graphs. Verify int32 map storage, offset zero and lookup consistency. Expect **48 output checks**.

No graph is replayed after its buffers move until a fresh capture is created. This does not claim that old graph pointers update automatically.'''),
code('moves=study.moves_gate()'),
md('''## 5. Update and persistence contracts

Reject wrong shape, wrong dtype and wrong-device updates before any mutation for all partitions. The wrong-device input is a meta tensor used only to trigger validation; it is never copied. Also reject enabling a map on a module constructed without mapping: **25 rejected invalid updates** total.

Check eight independent deep copies, named buffers, nonpersistent mapping exclusion and a marker-only checkpoint round trip. This does not test actual model checkpoint loading, meta-device model initialization or framework-wide buffer manipulation.'''),
code('contracts=study.contracts_gate()'),
md('''## 6. Export and remaining integration gates

Return the complete ZIP, including any failure evidence. CPU-only runs execute the matrix and update/persistence contracts, but skip CUDA update/move gates and therefore report an incomplete overall gate.

Even a passing CUDA run leaves these unvalidated: the actual small fused aligner/GEMMs, full serving/checkpoint integration, every production map writer's refresh discipline, concurrent-stream ordering, and H200/full-serving performance. Scheduler/tokenizer startup is outside this experiment.

There are no latency benchmarks or profiler traces here. Section wall times are execution diagnostics only. Passing this source-slice gate does not promote the draft.'''),
code('archive=study.export()\nfrom IPython.display import FileLink,display\ntry:\n    link=archive.relative_to(Path.cwd())\nexcept ValueError:\n    link=archive\ndisplay(FileLink(str(link)))'),
],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.10'}})
for c in nb.cells:
    if c.cell_type=='code':ast.parse(c.source)
for c in nb.cells[2:6]:c.metadata['jupyter']={'source_hidden':True}
nbformat.validate(nb);nbformat.write(nb,HERE/'12_kaggle_two_buffer_gate.ipynb')
print('Built',len(nb.cells),'cells')
