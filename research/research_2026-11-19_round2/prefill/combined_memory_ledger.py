"""Source lifetime accounting; not a serving allocator measurement."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVING = HERE.parents[1]
moe = json.loads((HERE.parent/'moe/hybrid_traffic_memory.json').read_text())['scratch']
W = 8519680  # Local Torch native workspace receipt; H200 must query its runtime.
K, N, TP = 40960, 8192, 8
KV = 128*1024*1024
MARGIN = 32*1024*1024
resident_moe = moe['combined_torch_owned_if_shared'] + moe['native_wgs_tiles']
row_per_m = 2*K + 2*N + 2*N//TP


def row_qualification(m):
    return dict(rows=m, synthetic_input=2*m*K, native_reference=2*m*N,
                local_output=2*m*N//TP, three_workspaces=3*W,
                margin=MARGIN, source_bound=m*row_per_m+3*W+MARGIN)


def allowed_max_rows(budget, other_resident):
    # Match existing generic TP divisibility, not a benchmark-specific bucket.
    rows=max(0,(budget-other_resident-3*W-MARGIN)//row_per_m)
    return rows//TP*TP


phases=[]
for kv_resident in (False,True):
    others=resident_moe+(KV if kv_resident else 0)
    for m in (2048,3072,4096,8192,16384):
        q=row_qualification(m)
        phases.append(dict(phase='row qualification after MoE arena',kv_resident=kv_resident,
                           other_candidate_resident=others,qualification=q,
                           total_incremental_source_bound=others+q['source_bound']))
out=dict(
    scope='Source/lifetime ledger, NOT measured whole-serving peak. Decimal2GB ceiling. Local workspace size only; query H200 runtime and all-rank-min budget.',
    baseline_common=dict(original_fc_weight=671088640,
                         note='Original checkpoint FC and five KV weights, target/draft weights, stock KV pools and stock captured graph pools are not charged again. During REAL projection the live input/native output sizes are baseline-common; during STARTUP qualification the generated input/reference are additional.'),
    sources=dict(row='research_2026-11-19/prefill/row_fc_runtime_helpers.py:create_prefill_projection, _row_startup_timing',
                 row_plan='research_2026-11-19/prefill/row_fc_adapter.py:discover, RowPlan.local, RowProjectionRouter',
                 n='research_2026-11-19/decode/output_n_package/entry.py,adapter.py,runtime_projection.py',
                 moe='research_2026-11-19_round2/moe/hybrid_traffic_memory.json'),
    formulas=dict(row_qualification='M*(2*K+2*N+2*N/TP)+3*W+32MiB',
                  output_n_qualification='M*(4*K+6*N+2*N/TP)+2*W <=128MiB',
                  row_runtime_extra_fastpath='local_output M*N*2/TP + one retained row workspace W; assembled output replaces same-size original FC output',
                  row_runtime_recoverable_fallback_extra='native output plus local slice plus gathered output coexist: conservatively M*N*2 + M*N*2/TP + W extra vs native FC',
                  n_runtime_fastpath_extra='rank-major gather intermediate M*N*2 +local M*N*2/TP +W; final gathered output replaces native output',
                  geometry_admission='floor_TP((combined_budget-other_resident-3*W-32MiB)/(2*K+2*N+2*N/TP))'),
    shared_moe_torch=moe['combined_torch_owned_if_shared'],
    shared_moe_native=moe['native_wgs_tiles'],resident_moe=resident_moe,
    kv_bound=KV,
    persistent_row_and_n_one_stream_each=2*W,
    persistent_sum_one_stream=resident_moe+KV+2*W,
    row_qualification_phases=phases,
    max_rows_by_combined_budget=[dict(budget=b,kv_resident=kv,other_resident=resident_moe+(KV if kv else 0),
                                          max_rows=allowed_max_rows(b,resident_moe+(KV if kv else 0)))
                                  for b in (1800000000,2000000000) for kv in (False,True)],
    lifecycle=[
        'Frozen joint worker qualifies row then N in prewarm_sampling AFTER initial KV pools and graph capture, BEFORE post_capture_resize. MoE scratch retained by those graphs cannot simply be freed around row qualification.',
        'KV diagnostic skips capture setup; default-disabled owns zero. First eligible live call currently owns one128MiB process-lifetime arena. Native FP8 gate and startup warmup unresolved.',
        'Row and N retain one vendor workspace per admitted stream/size, not per shape. Multiple streams add workspaces; no unbounded-stream assumption is permitted.',
        'Target/prefill graph pools remain stock-common only if candidate source proves no new graph-retained intermediates. BCG KV staging is eager and returns only native attention output, so its arena is not duplicated per layer/bucket.',
        'Real FC input/weights are baseline-common. New synthetic calibration input is not. Local fixture peak1.687GB must not be treated as net memory without separating these.',
        'Allocated and reserved are different. Freed qualification tensors can remain allocator-reserved; native CUDA handles, NCCL/symmetric buffers and graph-private retained blocks require actual allocator/cudaMemGetInfo comparison.',
    ],
    decision='UNMODIFIED joint530e rowM16384 qualification cannot combine with retained MoE arena under2GB. Need source-level global budget admission or qualification before arena construction/graph capture; neither change is applied here.',
)
(HERE/'combined_memory_ledger.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:out[k] for k in ('resident_moe','persistent_sum_one_stream','max_rows_by_combined_budget','decision')},indent=2))
