"""Index source boundaries for prefill prologue feasibility and startup lifetime."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
items=[
 ('frozen model decode/verify-only fused store','release_2026-11-18/candidate_sources/python/sglang/srt/models/muse_spark.py','Prefill excluded'),
 ('model original QKnorm/RoPE and cache-write selection','release_2026-11-18/candidate_sources/python/sglang/srt/models/muse_spark.py','        fuse_rope ='),
 ('backend-owned full/SWA locations','release_2026-11-18/candidate_sources/python/sglang/srt/layers/attention/flashattention_backend.py','    def get_fused_kv_write_buffers'),
 ('native cache append and FP8 attention dispatch','release_2026-11-18/candidate_sources/python/sglang/srt/layers/attention/flashattention_backend.py','    def forward_extend'),
 ('BCG argument identity and current metadata requirement','sglang/python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py','def eager_on_graph'),
 ('preallocated output and existing attention boundary','sglang/python/sglang/srt/layers/radix_attention.py','        use_eager_attention ='),
 ('current forward batch and real rows at eager attention','sglang/python/sglang/srt/layers/radix_attention.py','    def _eager_attention'),
 ('scoped cache metadata and optional arguments','sglang/python/sglang/srt/layers/attention/graph_utils.py','def attention_input_scope'),
 ('normalization original row/block policy','sglang/python/sglang/kernels/ops/attention/muse_spark_qk_norm_rope_store.py','def _triton_config'),
 ('normalization original BF16 and RoPE boundaries','sglang/python/sglang/kernels/ops/attention/muse_spark_qk_norm_rope_store.py','def _qk_rms_norm_fwd_kernel'),
 ('decoder-only prologue host eligibility','research_2026-11-19_round2/decode/prologue_package/host.py','def maybe_fuse'),
 ('decoder prologue shape calibration budget','research_2026-11-19_round2/decode/prologue_package/runtime.py','    def _qualify'),
 ('prologue original18-head math and K/V store','research_2026-11-19_round2/decode/prologue_package/kernel.py','def fusion18_kernel'),
 ('startup graph order','sglang/python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py','# cuda-graph capture: prefill before decode'),
 ('scheduler grad guard scope','sglang/python/sglang/srt/managers/scheduler.py','    @torch.no_grad()'),
 ('row factory missing no_grad in frozen joint','research_2026-11-19/prefill/integration_joint/sources/python/sglang/srt/speculative/dflash_projection_row.py','def create_prefill_projection'),
 ('draft initialized once','research_2026-11-19/prefill/integration_joint/sources/python/sglang/srt/speculative/dflash_worker_v2.py','    def _initialize_projection_adapters'),
 ('weight overlap rejects speculative and draft','sglang/python/sglang/srt/model_executor/model_runner_components/startup_weight_load.py','draft workers are not supported'),
]
records=[]
for topic,relative,needle in items:
 p=ROOT/relative;source=p.read_text();lines=source.splitlines()
 matches=[i+1 for i,line in enumerate(lines) if needle in line]
 assert matches,(relative,needle)
 records.append(dict(topic=topic,path=relative,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),needle=needle,lines=matches))
out=dict(scope='Local exact source index. Frozen baseline and separate unapplied decoder package, not deployed H200 binary provenance.',records=records)
(HERE/'prefill_guard_source_index.json').write_text(json.dumps(out,indent=2))
print('indexed',len(records),'boundaries')
