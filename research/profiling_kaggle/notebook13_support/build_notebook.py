import ast,hashlib,json
from pathlib import Path
import nbformat
HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\OpenKernelChallenge\serving');SOURCE=ROOT/'profiling_kaggle/notebook12_support'
files=['timing_support.py','source_slices.py','input_harness.py','gate_study.py','mapping_helper.py']
for n in files:(HERE/n).write_text((SOURCE/n).read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
manifest={'selected_draft_manifest':json.loads((SOURCE/'source_manifest_12.json').read_text()),
    'scope':'Unchanged two-buffer source slices; explicit event-ordered producer/consumer handoff on device 0. Correctness only.',
    'file_sha256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in files+['stream_study.py']}}
(HERE/'source_manifest_13.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
nb=nbformat.v4.new_notebook(cells=[
md('''# 13 - Order map updates before graph replay across streams

Notebook 12 passed the two-buffer source-slice correctness gate on a different Kaggle software stack. It tested updates and replay on an ordered stream. This notebook checks explicit event ordering when a separate producer stream updates the map and inputs.

**Kaggle: upload this file, select GPU T4, and Run All.** Only GPU 0 is used even if two GPUs are visible. No installation, Internet, model weights, datasets or credentials are needed. Keep the notebook private if you do not want to publish its embedded source slices.

Download `v4_stream_ordered_*.zip` into `profiling_kaggle/notebook13_results`. Export even if a section fails, then stop the GPU session.

No latency measurements are made. The selected two-buffer draft and active V4 remain unchanged; actual fused alignment/GEMMs, model loading and startup are not executed.'''),
md('''## 1. The handoff being tested

1. The producer waits for the prior consumer completion before overwriting shared inputs.
2. On the producer stream, overwrite old contents, copy new routing IDs, update maps and refresh lookup.
3. Record a **ready event** on the producer.
4. The consumer waits for ready, replays one graph containing baseline and candidate prefixes, and records **done**.
5. The host waits for done before checking outputs. The next producer cycle also waits for the previous done event.

There is no device-wide synchronization between submitting the update and submitting consumer replay. Setup and teardown may synchronize globally. Host validation waits after every cycle, so this is an ordered correctness test, not a throughput or fully overlapped multi-request benchmark.

No intentionally unsynchronized case is run: a race that happens to pass would not establish correctness. Production call sites must independently implement correct refresh and stream ordering.'''),
*[code((HERE/n).read_text(encoding='utf-8')) for n in files[:4]],
code((HERE/'stream_study.py').read_text(encoding='utf-8')),
code('SOURCE_MANIFEST='+repr(manifest)+'\nMAPPING_HELPER_SOURCE='+repr((HERE/'mapping_helper.py').read_text(encoding='utf-8'))+'\n\nstudy=StreamStudy(SOURCE_MANIFEST,MAPPING_HELPER_SOURCE)\nprint("Results directory:",study.out)'),
md('''## 2. Environment and prerequisites

Record the installed stack without modifying it. Check 16 baseline/candidate outputs across eight partitions, then re-run malformed-update, buffer ownership and marker-checkpoint contracts from Notebook 12.

CPU-only execution runs these prerequisites and exports results, but skips the CUDA handoff and reports the overall gate as incomplete. The large partition matrix from Notebook 12 is not repeated here.'''),
code('environment=study.environment()\nsmoke=study.smoke()\ncontracts=study.contracts_gate()'),
md('''## 3. Producer/consumer correctness gate

Cover eight partitions, two capture phases (initial and after GPU -> CPU -> GPU relocation), two consumer streams (default/dedicated), two update APIs (checked/explicit refresh), three rounds of changed routing IDs, and four mapping states (initial/all-remote/reversed/next partition).

Expected **768 handoff cycles, 1,536 output-value checks, 768 stable-pointer checks and eight recaptures after device moves**. All IDs are int32, scores float32, and inputs have 32 tokens with top-k eight. Every case uses the fallback mapping path; there are no metadata-only fused-branch checks in this gate.

Incoming IDs and maps already reside on GPU before the handoff loops and remain alive throughout. The producer deliberately overwrites old map/lookup entries before each update so a missed update cannot pass unnoticed. A ready-event dependency must precede replay. Completion events finish before validation and reuse. Pointer stability is checked within each capture phase, not across relocation.

Each case runs baseline and unchanged two-buffer source prefixes in the same captured graph; output values are checked against the independent Python reference. Stages are synthetic prefixes, not full model layers. No arbitrary concurrent map writers or cross-device operation is tested.'''),
code('handoff=study.handoff()'),
md('''## 4. Export

Return the complete ZIP. It contains per-cycle evidence, stream handles, pass/failure details, source provenance and checksums. Section wall times are diagnostics only, not latency results.

A passing gate establishes the tested explicit-event schedule. It does not establish that production map writers use it, that full serving/checkpoint integrations preserve both buffers, or that real fused kernels/model execution are correct. H200 performance and scheduler/tokenizer initialization remain outside scope. No draft is applied or promoted.'''),
code('archive=study.export()\nfrom IPython.display import FileLink,display\ntry:\n    link=archive.relative_to(Path.cwd())\nexcept ValueError:\n    link=archive\ndisplay(FileLink(str(link)))'),
],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.10'}})
for c in nb.cells:
    if c.cell_type=='code':ast.parse(c.source)
for c in nb.cells[2:7]:c.metadata['jupyter']={'source_hidden':True}
nbformat.validate(nb);nbformat.write(nb,HERE/'13_kaggle_stream_ordering.ipynb')
print('Built',len(nb.cells),'cells')
