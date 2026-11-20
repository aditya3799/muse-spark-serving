import ast,hashlib,json
from pathlib import Path
import nbformat
HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\OpenKernelChallenge\serving');SOURCE=ROOT/'profiling_kaggle/notebook09_support'
files=['timing_support.py','source_slices.py','input_harness.py','ownership_study.py','update_study.py','mapping_helper.py']
for n in files:(HERE/n).write_text((SOURCE/n).read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
audit=[]
for rel,fn in [('sglang/python/sglang/kernels/ops/moe/muse_spark_routing.py','fused_muse_spark_routing'),('sglang/python/sglang/kernels/ops/moe/muse_spark_routing_jit.py','fused_muse_spark_routing_jit')]:
    content=(ROOT/rel).read_text(encoding='utf-8');tree=ast.parse(content)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==fn)
    allocations=[]
    for n in ast.walk(node):
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('topk_weights','topk_ids') and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='torch.empty':
            dtype=next(ast.unparse(k.value) for k in n.value.keywords if k.arg=='dtype')
            allocations.append({'variable':n.targets[0].id,'dtype':dtype,'line':n.lineno,'source':ast.get_source_segment(content,n)})
    assert {a['variable']:a['dtype'] for a in allocations}=={'topk_weights':'torch.float32','topk_ids':'torch.int32'}
    audit.append({'path':rel,'sha256':hashlib.sha256(content.encode()).hexdigest(),'function':fn,'default_allocations':allocations})
model_rel='sglang/python/sglang/srt/models/muse_spark.py';model=(ROOT/model_rel).read_text(encoding='utf-8')
evidence={'router_allocations':audit,'model_source':{'path':model_rel,'sha256':hashlib.sha256(model.encode()).hexdigest(),'call_lines':[{'line':i,'source':s.strip()} for i,s in enumerate(model.splitlines(),1) if 'topk_scores, topk_ids, packed_topk_ids = self._routing_op()' in s]},
    'limits':'Default allocations only. Actual backend, provided output buffers/layout and model runtime have not been observed. Router kernels are not embedded or executed.'}
(HERE/'ROUTER_CONTRACT_AUDIT.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
manifest={'parent_source_manifest':json.loads((SOURCE/'source_manifest_09.json').read_text()),'router_contract_audit':evidence,
    'scope':'Unchanged source prefixes with synthetic routing inputs of four dtype combinations. Default router contract tested without executing router.',
    'file_sha256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in files+['contract_study.py']}}
(HERE/'source_manifest_10.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
nb=nbformat.v4.new_notebook(cells=[
md('''# 10 - Match the router's default output formats

Notebook 09 showed mapping gains inside combined graphs. A source audit found a representativeness gap: that harness used int64 IDs and float16 scores, while the inspected JIT and Triton router defaults allocate int32 IDs and float32 scores. This experiment isolates the resulting conversion work.

**Kaggle: upload this file, select GPU T4, and Run All.** Only GPU 0 is used. No installation, Internet, datasets, model weights or credentials are needed. Keep it private if you do not want to publish the embedded project source slices.

Download `v4_router_contract_*.zip` into `profiling_kaggle/notebook10_results`. Export even after an earlier section failure, then stop the GPU session.

Scope is still **runtime MoE preparation**. Actual router, alignment, expert GEMM, full model and startup are not executed. Active V4 and both drafts remain unchanged.'''),
md('''## 1. Four input contracts, one unchanged mapping implementation

| Contract | Expert IDs | Scores | Purpose |
|---|---|---|---|
| `i64_s16` | int64 | float16 | Historical harness control; both conversions needed |
| `i32_s16` | int32 | float16 | Remove ID conversion only |
| `i64_s32` | int64 | float32 | Remove score conversion only |
| `i32_s32` | int32 | float32 | Match inspected default router output types |

For each contract, compare baseline and single-owner prefixes. All eight variants are shuffled within every timing sample. The router's default allocation types are verified from local source by the notebook builder and recorded with source hashes; the actual router is not called here. Preallocated buffers, runtime layout and selected backend remain unobserved.

All inputs are contiguous during timing. Scores equal 0.125 in every format, so changing dtype does not change their numeric values. Hidden activations remain synthetic float16 tensors of width 32; this prefix only uses their metadata and contiguity. Each independent stage owns a distinct static mapping; the stages do not consume each other's outputs.'''),
*[code((HERE/n).read_text(encoding='utf-8')) for n in files[:5]],
code((HERE/'contract_study.py').read_text(encoding='utf-8')),
code('SOURCE_MANIFEST='+repr(manifest)+'\nMAPPING_HELPER_SOURCE='+repr((HERE/'mapping_helper.py').read_text(encoding='utf-8'))+'\n\nSAMPLES=11\nREPEATS=3\nTRIALS=3\nstudy=ContractStudy(SOURCE_MANIFEST,MAPPING_HELPER_SOURCE,samples=SAMPLES,repeats=REPEATS,trials=TRIALS)\nprint("Results directory:",study.out)'),
md('''## 2. Correctness gates

Re-run the inherited lifecycle gate (192 configurations / 576 outputs). The new input-contract gate checks **40 configurations / 80 outputs** across all four dtype pairs, 0/1/8/9/32 tokens and contiguous/strided ID inputs. It checks score-storage reuse when scores are already float32.

Small fused-branch outputs are checked only for branch choice and metadata; their uninitialized ID values are never read. The real fused kernel remains untested. Strided layouts and empty inputs are correctness controls, not timed scenarios. CPU-only execution runs these gates and skips CUDA timing.'''),
code('environment=study.environment()\nlifecycle=study.lifecycle()\ninput_contract=study.input_contract()'),
md('''## 3. Interleaved combined-graph measurements

**9/32/128 tokens**, **1/8 independent stages**, three shuffled trials: **18 cases**. Eight variants per case, eleven samples per variant, three complete chain replays per sample: **144 timing rows and 1,584 raw samples**.

Before timing, change routing ID values twice and validate every stage in every variant: **1,296 changed-input output checks**. Scores, input shapes and mapping contents stay fixed during timing; all outputs are checked again afterward. Mapping setup, captures and validation occur outside timing.

CUDA-event time, synchronized wall time and host enqueue time are reported per complete chain. These intervals overlap and must not be added or subtracted to infer exact CPU/GPU shares. Per-stage values are chain time divided by stage count. All eight variants are interleaved within the same sample to reduce drift across input contracts.'''),
code('benchmark=study.benchmark()'),
md('''## 4. Separate trace

Trace all eight variants at 128 tokens and eight stages, three cycles each. Check how actual kernel counts change when ID or score conversions become unnecessary. Annotation spans are excluded. Missing CUDA trace support or graph correlation produces partial status.'''),
code('profiler=study.profile()'),
md('''## 5. Export and interpretation

The ZIP includes raw samples, all timing metrics, **252 comparisons**, lifecycle/input-contract evidence, source audit, trace and checksums. Seventy-two comparisons measure candidate versus baseline within each contract; the other 180 compare dtype controls within an implementation.

The main result is candidate versus baseline for `i32_s32`. Dtype-control gains quantify differences in the synthetic harness, not a newly implemented production optimization. Prior comparisons remain valid for their tested inputs, but should not be presented as timings for the router's default output types.

There is still no full-serving/H200 speedup, actual router/GEMM/alignment validation, framework-wide buffer compatibility result, or scheduler/tokenizer initialization measurement. Neither draft is promoted.'''),
code('archive=study.export()\nfrom IPython.display import FileLink,display\ntry:\n    link=archive.relative_to(Path.cwd())\nexcept ValueError:\n    link=archive\ndisplay(FileLink(str(link)))'),
],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.10'}})
for c in nb.cells:
    if c.cell_type=='code':ast.parse(c.source)
for c in nb.cells[2:8]:c.metadata['jupyter']={'source_hidden':True}
nbformat.validate(nb);nbformat.write(nb,HERE/'10_kaggle_router_contract.ipynb')
print('Built',len(nb.cells),'cells')
